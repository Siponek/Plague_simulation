#!/usr/bin/env python

import argparse
import configparser
import logging
import random
from abc import abstractmethod
from dataclasses import dataclass
from random import expovariate
from typing import Optional, List
from matplotlib import pyplot as plt

# the humanfriendly library (https://humanfriendly.readthedocs.io/en/latest/) lets us pass parameters in human-readable
# format (e.g., "500 KiB" or "5 days"). You can safely remove this if you don't want to install it on your system, but
# then you'll need to handle sizes in bytes and time spans in seconds--or write your own alternative.
# It should be trivial to install (e.g., apt install python3-humanfriendly or conda/pip install humanfriendly).
from humanfriendly import format_timespan, parse_size, parse_timespan

from discrete_event_sim_redacted import Simulation, Event


def exp_rv(mean):
    """Return an exponential random variable with the given mean."""
    return expovariate(1 / mean)


class DataLost(Exception):
    """Not enough redundancy in the system, data is lost. We raise this exception to stop the simulation."""
    pass


class Backup(Simulation):
    """Backup simulation.
    """

    # TODO: parallel uploads/downloads DONE -> PROS: easy to implement,
    #                                         obvious optimalization, system speedup, cost effective;
    #                                         CONS: not sofisticated
    # selfish backup behaviour (first me,
    # then the others) -> PROS: ease of implementation;
    #                     CONS: don't be greedy, Does not improve the system in noticeable way
    # tit for tat (peers ranked basing
    # upon the number of "good actions"
    # made to each other) -> PROS: prefers "stronger" nodes / more backup on nodes that fail less,
    #                              way of analysing network?;
    #                        CONS: lost of equality between nodes and reliability, not distributed data,
    #

    # type annotations for `Node` are strings here to allow a forward declaration:
    # https://stackoverflow.com/questions/36193540/self-reference-or-forward-reference-of-type-annotations-in-python
    def __init__(self, nodes: List['Node'], parallel_up_down: bool, fails, fails_after_recover,
                 onlines, offlines, recovers):
        super().__init__()  # call the __init__ method of parent class
        self.nodes = nodes

        # flag that allows nodes to perform multiple downloads and uploads at the same time
        self.parallel_up_down = parallel_up_down
        # self.max_node_dw = 0
        self.dw_bw_wasted = {}  # map time - average download bandwidth wasted by the nodes
        #self.max_node_up = 0
        self.up_bw_wasted = {}  # map time - average upload bandwidth wasted by the nodes

        # map of preestablished events used for simulation purpose
        # (we want to compare how the normal and the extended system behave
        # under the same circumstances)
        self.fails = fails  # first fail delay  for each node
        self.offlines = offlines  # offline events' delay
        self.fails_after_recover = fails_after_recover  # next fail after recover events' delay
        self.onlines = onlines  # online events' delay after disconnection
        self.recovers = recovers  # recover events' delay after failure

        # we add to the event queue the first event for each node going online and failing
        for node in nodes:
            self.schedule(node.arrival_time, Online(node))
            self.schedule(node.arrival_time + self.fails[node], Fail(node))

    # returns and removes the head of the list and behaves like a circular buffer
    def pop_time(self, l):
        time = l.pop(0)
        l.append(time)
        return time

    # registers bandwidth waste
    def register_bw_waste(self, time):

        # we take into account only those nodes that are online and that are uploading something
        l_up = [node.available_bw_upload for node in self.nodes if node.online
                and len(node.current_uploads.values()) > 0]

        # we take into account only those nodes that are online and that are downloading something
        l_down = [node.available_bw_download for node in self.nodes if node.online
                and len(node.current_downloads.values()) > 0]

        # registering average upload/download bandwidth waste for given time
        self.up_bw_wasted[time] = sum(l_up) / len(l_up) if len(l_up) > 0 else 0
        self.dw_bw_wasted[time] = sum(l_down) / len(l_down) if len(l_down) > 0 else 0

    # plotting
    def plot_wasted_bw(self, plt):

        points1 = list(self.up_bw_wasted.keys())
        values1 = self.up_bw_wasted.values()

        points2 = list(self.dw_bw_wasted.keys())
        values2 = self.dw_bw_wasted.values()

        t = "single" if not self.parallel_up_down else "parallel"
        plt.suptitle("Average wasted bandwidth over time with " + t +
                     " uploads and downloads")

        plt.subplot(1, 2, 1)
        plt.title("Average wasted upload bandwidth")
        plt.plot(points1, values1)
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.title("Average wasted download bandwidth")
        plt.plot(points2, values2)

        plt.grid()
        # plt.show()

    def schedule_transfer(self, uploader: 'Node', downloader: 'Node', block_id: int, restore: bool):
        """Helper function called by `Node.schedule_next_upload` and `Node.schedule_next_download`.

        If `restore` is true, we are restoring a block owned by the downloader, otherwise, we are saving one owned by
        the uploader.
        """

        # the block size depends on the direction of the transfer:
        # if it is a restore (from uploader to downloader), we take
        # the downloader's block size; otherwise, if it is a backup,
        # we use the uploader's one
        block_size = downloader.block_size if restore else uploader.block_size

        # ensure that both uploader and downloader have some bandwidth available
        # otherwise, it is an error
        assert uploader.available_bw_upload > 0
        assert downloader.available_bw_download > 0

        # we take the slowest between the available bandwidths
        speed = min(uploader.available_bw_upload, downloader.available_bw_download)

        # calculate the time needed for the transfer
        delay = block_size / speed

        # update available bandwidth for uploader and downloader
        uploader.available_bw_upload -= speed
        downloader.available_bw_download -= speed

        # distinguish restore from backup transfer
        if restore:
            event = BlockRestoreComplete(uploader, downloader, block_id, speed)
        else:
            event = BlockBackupComplete(uploader, downloader, block_id, speed)

        # schedule the completion event with the calculated delay
        # and add a new ongoing transfer instance in uploader's
        # current uploads and in the downloader's current downloads
        self.schedule(delay, event)
        uploader.current_uploads[(uploader, downloader, block_id)] = event
        downloader.current_downloads[(uploader, downloader, block_id)] = event

    def log_info(self, msg):
        """Override method to get human-friendly logging for time."""
        logging.info(f'{format_timespan(self.t)}: {msg}')


class BackupSim:

    def __init__(self, max_t, nodes):
        self.fails = {}

        # calculating needed mean
        avg_lt = sum([node.average_lifetime for node in nodes]) / len(nodes)
        avg_dt = sum([node.average_downtime for node in nodes]) / len(nodes)
        avg_rt = sum([node.average_recover_time for node in nodes]) / len(nodes)

        # fails occurs with avergae lifetime as the mean
        for node in nodes:
            self.fails[node] = exp_rv(node.average_lifetime)

        # offlines occur with avergae lifetime as the mean
        self.offlines = self.init_list(max_t, avg_lt)
        # fails occur with avergae lifetime as the mean
        self.fails_after_recover = self.init_list(max_t, avg_lt)
        # onlines occur with avergae downtime as the mean
        self.onlines = self.init_list(max_t, avg_dt)
        # recovers occur with avergae recover time as the mean
        self.recovers = self.init_list(max_t, avg_rt)

        # creating and launching different simulations with the same circumstances
        normal_sim = Backup(nodes, False, self.fails, self.fails_after_recover, self.onlines, self.offlines,
                            self.recovers)
        extended_sim = Backup(nodes, True, self.fails, self.fails_after_recover, self.onlines, self.offlines,
                            self.recovers)
        normal_sim.run(max_t)
        extended_sim.run(max_t)

        # plotting results
        points1 = list(normal_sim.up_bw_wasted.keys())
        values1 = [val / pow(10, 6) for val in normal_sim.up_bw_wasted.values()]  # bandwidth is represented in MiB

        points2 = list(normal_sim.dw_bw_wasted.keys())
        values2 = [val / pow(10, 6) for val in normal_sim.dw_bw_wasted.values()]

        points3 = list(extended_sim.up_bw_wasted.keys())
        values3 = [val / pow(10, 6) for val in extended_sim.up_bw_wasted.values()]

        points4 = list(extended_sim.dw_bw_wasted.keys())
        values4 = [val / pow(10, 6) for val in extended_sim.dw_bw_wasted.values()]

        fig, axs = plt.subplots(2, 2)
        plt.suptitle("Average wasted bandwidth over time")
        axs[0, 0].plot(points1, values1)
        axs[0, 0].set_title('Single upload bandwidth wasted')
        axs[0, 1].plot(points2, values2, 'tab:orange')
        axs[0, 1].set_title('Single download bandwidth wasted')
        axs[1, 0].plot(points3, values3, 'tab:green')
        axs[1, 0].set_title('Parallel upload bandwidth wasted')
        axs[1, 1].plot(points4, values4, 'tab:red')
        axs[1, 1].set_title('Parallel download bandwidth wasted')

        for ax in axs.flat:
            ax.set(xlabel='Time', ylabel='Average wasted bandwidth (MB)')
        fig.tight_layout()
        plt.show()

    # utility that returns a list of random values having the specified mean generated with exponential distribution
    def init_list(self, max_t, mean):
        act_t = 0
        l = []

        while act_t < max_t:
            new_t = exp_rv(mean)
            l.append(new_t)
            act_t += new_t
        return l


@dataclass(eq=False)  # auto initialization from parameters below (won't consider two nodes with same state as equal)
class Node:
    """Class representing the configuration of a given node."""

    # using dataclass is (for our purposes) equivalent to having something like
    # def __init__(self, description, n, k, ...):
    #     self.n = n
    #     self.k = k
    #     ...
    #     self.__post_init__()  # if the method exists

    name: str  # the node's name

    n: int  # number of blocks in which the data is encoded
    k: int  # number of blocks sufficient to recover the whole node's data

    data_size: int  # amount of data to back up (in bytes)
    storage_size: int  # storage space devoted to storing remote data (in bytes)

    upload_speed: float  # node's upload speed, in bytes per second
    download_speed: float  # download speed

    average_uptime: float  # average time spent online
    average_downtime: float  # average time spent offline

    average_lifetime: float  # average time before a crash and data loss
    average_recover_time: float  # average time after a data loss

    arrival_time: float  # time at which the node will come online

    def __post_init__(self):
        """Compute other data dependent on config values and set up initial state."""

        # whether this node is online. All nodes start offline.
        self.online: bool = False

        # whether this node is currently under repairs. All nodes are ok at start.
        self.failed: bool = False

        # size of each block
        self.block_size: int = self.data_size // self.k if self.k > 0 else 0

        # amount of free space for others' data -- note we always leave enough space for our n blocks
        self.free_space: int = self.storage_size - self.block_size * self.n

        assert self.free_space >= 0, "Node without enough space to hold its own data"

        # local_blocks[block_id] is true if we locally have the local block
        # [x] * n is a list with n references to the object x
        self.local_blocks: list[bool] = [True] * self.n

        # backed_up_blocks[block_id] is the peer we're storing that block on, or None if it's not backed up yet;
        # we start with no blocks backed up
        self.backed_up_blocks: list[Optional[Node]] = [None] * self.n

        # (owner -> block_id) mapping for remote blocks stored
        self.remote_blocks_held: dict[Node, int] = {}

        # bandwidth available for upload, initially set as upload speed
        self.available_bw_upload: float = self.upload_speed

        # bandwidth available for download, initially set as download speed
        self.available_bw_download: float = self.download_speed

        # current uploads and downloads, stored as a reference to the relative TransferComplete event
        # each transfer is identified through uploader, downloader and block_id
        self.current_uploads: dict[(Node, Node, int), TransferComplete] = {}
        self.current_downloads: dict[(Node, Node, int), TransferComplete] = {}

    def find_block_to_back_up(self):
        """Returns the block id of a block that needs backing up, or None if there are none."""

        # find a block that we have locally but not remotely
        # check `enumerate` and `zip`at https://docs.python.org/3/library/functions.html
        for block_id, (held_locally, peer) in enumerate(zip(self.local_blocks, self.backed_up_blocks)):
            if held_locally and peer is None:  # block_id not present locally and no one has it backed up yet
                return block_id
        return None

    # returns true when a transfer gets scheduled, false otherwise
    def schedule_next_upload(self, sim: Backup):
        """Schedule the next upload, if any."""

        # in order to try to upload something, i have to be online,
        # otherwise it is an error
        assert self.online

        # if i don't have available bandiwidth for upload, i can't perform it
        if self.available_bw_upload == 0 or (not sim.parallel_up_down and len(list(self.current_uploads.keys())) > 0):
            return False

        # first find if we have a backup that a remote node needs
        for peer, block_id in self.remote_blocks_held.items():

            if peer is not None:
                # if the peer is allowed to do parallel uploads/downloads then
                # the constraint is that the peer has available download bandwidth,
                # otherwise is that the peer is not downloading anything at the moment
                constraint = peer.available_bw_download > 0 if sim.parallel_up_down \
                    else len(list(peer.current_downloads.keys())) < 1
            else:  # no constraint on the peer
                constraint = True

            # if the block is not present locally and the peer is online and
            # the constraint is verified, then schedule the restore from self to peer of block_id
            if peer.online and constraint and not peer.local_blocks[block_id]:
                sim.schedule_transfer(self, peer, block_id, True)
                return True  # we have found our upload, we stop

        # try to back up a block locally held on a remote node
        block_id = self.find_block_to_back_up()

        # if i can't find a block to back up, then i'm done
        if block_id is None:
            return False

        # nodes having one of my blocks
        remote_owners = set(node for node in self.backed_up_blocks if node is not None)
        for peer in sim.nodes:
            if peer is not None:
                # if the peer is allowed to do parallel uploads/downloads then
                # the constraint is that the peer has available download bandwidth,
                # otherwise is that the peer is not downloading anything at the moment
                constraint = peer.available_bw_download > 0 if sim.parallel_up_down \
                    else len(list(peer.current_downloads.keys())) < 1
            else:  # no constraint on the peer
                constraint = True

            # if the peer is not self, is online, is not among the remote owners, has enough space and the
            # constraint is verified, schedule the backup of block_id from self to peer
            if (peer is not self and peer.online and peer not in remote_owners and constraint
                    and peer.free_space >= self.block_size):
                sim.schedule_transfer(self, peer, block_id, False)  # scheduling the uploading of the block by this node
                return True                                         # and the downloading from the peer
        return False  # didn't find either a backup or a block to restore

    def schedule_next_uploads(self, sim: Backup):
        counter = 0

        # try to schedule a new upload until it is possible to avoid
        # waste of bandwidth (two scenarios to stop this task: no
        # blocks found for backup or restore; no more bandwidth available)
        while counter < 1 or sim.parallel_up_down:
            if not self.schedule_next_upload(sim):
                break
            counter = counter + 1

        #sim.max_node_up = len(self.current_uploads.values()) if len(self.current_uploads.values()) > sim.max_node_up \
        #    else sim.max_node_up
        sim.log_info(self.name + " scheduled " + str(counter) + " new uploads")  # logs new uploads scheduled
        sim.log_info(self.name + " is executing " + str(len(self.current_uploads.values())) + " uploads")

    def schedule_next_download(self, sim: Backup):
        """Schedule the next download, if any."""

        # in order to try to download something, i have to be online,
        # otherwise it is an error
        assert self.online

        # if i don't have available bandiwidth for download, i can't perform it
        if self.available_bw_download == 0 or (not sim.parallel_up_down and len(list(self.current_downloads.keys())) > 0):
            return False

        # first find if we have a missing block to restore
        for block_id, (held_locally, peer) in enumerate(zip(self.local_blocks, self.backed_up_blocks)):
            if peer is not None:
                # if the peer is allowed to do parallel uploads/downloads then
                # the constraint is that the peer has available upload bandwidth,
                # otherwise is that the peer is not uploading anything at the moment
                constraint = peer.available_bw_upload > 0 if sim.parallel_up_down \
                    else len(list(peer.current_uploads.keys())) < 1
            else:  # no constraint on the peer
                constraint = True

            # if i don't have one of my blocks locally, but a peer does, if it is online
            # and the constraint is verified, then schedule the restore from peer to self
            if not held_locally and peer is not None and peer.online and constraint:
                sim.schedule_transfer(peer, self, block_id, True)
                return True  # we are done in this case

        # try to back up a block for a remote node
        for peer in sim.nodes:
            if peer is not None:
                # if the peer is allowed to do parallel uploads/downloads then
                # the constraint is that the peer has available upload bandwidth,
                # otherwise is that the peer is not uploading anything at the moment
                constraint = peer.available_bw_upload > 0 if sim.parallel_up_down \
                    else len(list(peer.current_uploads.keys())) < 1
            else:  # no constraint on the peer
                constraint = True

            # if the peer is not self and it is online and the constraint is verified
            # and self has not backed up anything for peer yet and has enough space, then look for a peer's
            # block that needs a backup: if you find it, then schedule the backup of that block from self
            # to peer
            if (peer is not self and peer.online and constraint and peer not in
                    self.remote_blocks_held.keys()
                    and self.free_space >= peer.block_size):
                block_id = peer.find_block_to_back_up()
                if block_id is not None:
                    sim.schedule_transfer(peer, self, block_id, False)
                    return True
        return False  # did not find a download to perform

    def schedule_next_downloads(self, sim: Backup):
        counter = 0

        # try to schedule a new download until it is possible to avoid
        # waste of bandwidth (two scenarios to stop this task: no
        # blocks found for backup or restore; no more bandwidth available)
        while counter < 1 or sim.parallel_up_down:
            if not self.schedule_next_download(sim):
                break
            counter = counter + 1

        #sim.max_node_dw = len(self.current_downloads.values()) if len(self.current_downloads.values()) > sim.max_node_dw \
        #    else sim.max_node_dw
        sim.log_info(self.name + " scheduled " + str(counter) + " new downloads")  # logs new downloads scheduled
        sim.log_info(self.name + " is executing " + str(len(self.current_downloads.values())) + " downloads")


    def __hash__(self):
        """Function that allows us to have `Node`s as dictionary keys or set items.

        With this implementation, each node is only equal to itself.
        """
        return id(self)

    def __str__(self):
        """Function that will be called when converting this to a string (e.g., when logging or printing)."""

        return self.name


@dataclass
class NodeEvent(Event):
    """An event regarding a node. Carries the identifier, i.e., the node's index in `Backup.nodes_config`"""

    node: Node  # target node of the event

    def __post_init__(self):
        super().__init__()

    @abstractmethod
    def process(self, sim: Simulation):
        """Must be implemented by subclasses."""
        pass


class Online(NodeEvent):
    """A node goes online."""

    def process(self, sim: Backup):
        node = self.node
        # if the node is already online or if it is failed,
        # then return
        if node.online or node.failed:
            return
        # when a node returns online, it has all bandwidth available
        node.online = True
        node.available_bw_upload = self.node.upload_speed
        node.available_bw_download = self.node.download_speed
        # schedule next uploads and downloads
        node.schedule_next_uploads(sim)
        node.schedule_next_downloads(sim)
        sim.register_bw_waste(sim.t)
        # schedule the next offline event
        delay = sim.pop_time(sim.offlines)
        sim.schedule(delay, Offline(node))


class Recover(Online):
    """A node goes online after recovering from a failure."""

    def process(self, sim: Backup):
        node = self.node
        sim.log_info(f"{node} recovers")

        # when a node recovers, it is not in the fail state no more
        node.failed = False

        # when a node recovers it has its own data, but not the blocks
        # belonging to others
        self.node.free_space = self.node.storage_size - self.node.block_size * self.node.n

        # recover includes returning in the online state
        super().process(sim)

        # schedule next fail
        delay = sim.pop_time(sim.fails_after_recover)
        sim.schedule(delay, Fail(node))


class Disconnection(NodeEvent):
    """Base class for both Offline and Fail, events that make a node disconnect."""

    @abstractmethod
    def process(self, sim: Simulation):
        """Must be implemented by subclasses."""
        pass

    def disconnect(self):
        node = self.node

        # when a node disconnects it is not online no more, so
        # it has no bandwidth available
        node.online = False
        node.available_bw_upload = 0
        node.available_bw_download = 0

        current_uploads = node.current_uploads
        current_downloads = node.current_downloads

        # cancel current uploads
        for transfer in current_uploads.values():
            if transfer is not None:
                # cancel the transfer
                transfer.canceled = True
                # remove the transfer from downloader's current downloads
                del transfer.downloader.current_downloads[(transfer.uploader, transfer.downloader, transfer.block_id)]

        # clear node's current uploads
        node.current_uploads = {}

        # cancel current downloads
        for transfer in current_downloads.values():
            if transfer is not None:
                # cancel the transfer
                transfer.canceled = True
                # remove the transfer from uploader's current uploads
                del transfer.uploader.current_uploads[(transfer.uploader, transfer.downloader, transfer.block_id)]

        # clear node's current downloads
        node.current_downloads = {}


class Offline(Disconnection):
    """A node goes offline."""

    def process(self, sim: Backup):
        node = self.node
        # if the node is already offline or if it is failed,
        # then return
        if node.failed or not node.online:
            return
        assert node.online
        # disconnect the node
        self.disconnect()
        # schedule the next online event
        delay = sim.pop_time(sim.onlines)
        sim.schedule(delay, Online(node))


class Fail(Disconnection):
    """A node fails and loses all local data."""

    def process(self, sim: Backup):
        sim.log_info(f"{self.node} fails")
        # disconnect the node
        self.disconnect()
        node = self.node
        node.failed = True
        # lose all local data
        node.local_blocks = [False] * node.n
        # lose all remote data
        for owner, block_id in node.remote_blocks_held.items():
            if owner is not None:
                # if the peer is allowed to do parallel uploads/downloads then
                # the constraint is that the peer has available upload bandwidth,
                # otherwise is that the peer is not uploading anything at the moment
                constraint = owner.available_bw_upload > 0 if sim.parallel_up_down \
                    else len(list(owner.current_uploads.keys())) < 1
            else:  # no constraint on the peer
                constraint = True

            owner.backed_up_blocks[block_id] = None

            # if the owner of the block that i lost is online
            # and the constraint is verified, try to upload the block somewhere else
            if owner.online and constraint:
                owner.schedule_next_uploads(sim)  # this node may want to back up the missing block
                sim.register_bw_waste(sim.t)

        # clearing other's block data
        node.remote_blocks_held.clear()
        # schedule the next recover event
        recover_time = sim.pop_time(sim.recovers)
        sim.schedule(recover_time, Recover(node))


@dataclass
class TransferComplete(Event):
    """An upload is completed."""

    uploader: Node  # uploader of the transfer
    downloader: Node  # downloader of the transfer
    block_id: int  # block object of the transfer
    speed: float  # bandwidth used by the peers for this transfer
    canceled: bool = False

    def __post_init__(self):

        # can not schedule a self download or upload
        assert self.uploader is not self.downloader

    def process(self, sim: Backup):
        sim.log_info(f"{self.__class__.__name__} from {self.uploader} to {self.downloader}")

        # this transfer was canceled, so ignore this event
        if self.canceled:
            return

        uploader, downloader = self.uploader, self.downloader

        # uploader and downloader must be online in order to complete the transfer,
        # otherwise it is an error
        assert uploader.online and downloader.online

        # updating block state
        self.update_block_state()

        # once the transfer is finished, bandwidth used is given back to peers
        uploader.available_bw_upload += self.speed
        downloader.available_bw_download += self.speed

        # this transfer is completed so i remove it from ongoing uploads/downloads of the peers involved
        if (self.uploader, self.downloader, self.block_id) in uploader.current_uploads.keys():
            del uploader.current_uploads[(self.uploader, self.downloader, self.block_id)]
        if (self.uploader, self.downloader, self.block_id) in downloader.current_downloads.keys():
            del downloader.current_downloads[(self.uploader, self.downloader, self.block_id)]

        # schedule next uploads/downloads for uploader/downloader
        uploader.schedule_next_uploads(sim)
        downloader.schedule_next_downloads(sim)
        sim.register_bw_waste(sim.t)
        for node in [uploader, downloader]:
            sim.log_info(f"{node}: {sum(node.local_blocks)} local blocks, "
                         f"{sum(peer is not None for peer in node.backed_up_blocks)} backed up blocks, "
                         f"{len(node.remote_blocks_held)} remote blocks held")

    @abstractmethod
    def update_block_state(self):
        """Needs to be specified by the subclasses, `BackupComplete` and `DownloadComplete`."""
        pass


class BlockBackupComplete(TransferComplete):

    def update_block_state(self):
        owner, peer = self.uploader, self.downloader
        peer.free_space -= owner.block_size  # updating free space after saving a block
        assert peer.free_space >= 0
        owner.backed_up_blocks[self.block_id] = peer  # mark peer as the one where block_id has been backed up

        # mark block_id as the block belonging to the owner that the peers stores
        peer.remote_blocks_held[owner] = self.block_id


class BlockRestoreComplete(TransferComplete):
    def update_block_state(self):
        owner = self.downloader
        owner.local_blocks[self.block_id] = True  # updating blocks held locally
        if sum(owner.local_blocks) == owner.k:  # we have exactly k local blocks, we have all of them then
            owner.local_blocks = [True] * owner.n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="configuration file", default="p2p.cfg")
    parser.add_argument("--max-t", default="50 years")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true', default=True)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout

    # functions to parse every parameter of peer configuration
    parsing_functions = [
        ('n', int), ('k', int),
        ('data_size', parse_size), ('storage_size', parse_size),
        ('upload_speed', parse_size), ('download_speed', parse_size),
        ('average_uptime', parse_timespan), ('average_downtime', parse_timespan),
        ('average_lifetime', parse_timespan), ('average_recover_time', parse_timespan),
        ('arrival_time', parse_timespan)
    ]

    config = configparser.ConfigParser()
    config.read(args.config)
    nodes = []  # we build the list of nodes to pass to the Backup class
    for node_class in config.sections():
        class_config = config[node_class]
        # list comprehension: https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
        cfg = [parse(class_config[name]) for name, parse in parsing_functions]
        # the `callable(p1, p2, *args)` idiom is equivalent to `callable(p1, p2, args[0], args[1], ...)
        nodes.extend(Node(f"{node_class}-{i}", *cfg) for i in range(class_config.getint('number')))
    sims = BackupSim(parse_timespan(args.max_t), nodes)
    # sim = Backup(nodes, args.extended)
    # sim.run(parse_timespan(args.max_t))


if __name__ == '__main__':
    main()
