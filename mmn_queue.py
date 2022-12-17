##!/usr/bin/env python

import argparse
import csv
import collections
from random import expovariate, choice, sample

from discrete_event_sim import Simulation, Event

# ? can we use random.expovariate() here?
from workloads import weibull_generator

# from reliability.Distributions import Weibull_Distribution
from functools import partial

# To use weibull variates, for a given set of parameter do something like
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


class MMN(Simulation):
    def __init__(self, lambd, mu, n):
        # extend this to make it work for multiple queues
        # if n != 1:
        #     raise NotImplementedError

        super().__init__()
        # if not None, the id of the running job
        self.running: list = [None for _ in range(n)]
        # FIFO queue of the system
        self.queue = [collections.deque() for _ in range(n)]
        # dictionary mapping job id to arrival time
        self.arrivals = {}
        # dictionary mapping job id to completion time
        self.completions = {}
        # time waiting in the queue
        self.lambd = lambd
        # number of queues
        self.n = n
        # time spent in the system
        self.mu = mu
        self.arrival_rate = lambd / n
        self.completion_rate = mu / n
        self.supermarket_scheduler: bool = False
        # self.schedule(expovariate(lambd=lambd), Arrival(0))

    @property
    def queue_len(self):
        # todo: this is not correct, it should be the sum of the queue lengths
        # todo it needs to return list of length of queues
        return [True for x in self.running if x is None] + [
            len(x) for x in self.queue
        ]
        # return (self.running[0] is None) + len(self.queue)

    def schedule_arrival(self, job_id: int):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        # Rate of arrival will be lambda*n
        self.schedule(expovariate(self.lambd * self.n), Arrival(job_id))

    def schedule_completion(self, job_id: int):
        # schedule the time of the completion event
        self.schedule(expovariate(lambd=self.mu), Completion(job_id))

    # Supermarket queue:
    def supermarket_choice(self):
        sample_of_queues = sample(population=range(self.n), k=self.sample_size)

        if len(sample_of_queues) == 1:
            return sample_of_queues[0]
        minQueue: collections.deque = sample_of_queues[0]

        # ? Comparing ?
        for queue in sample_of_queues:
            # If the length of the queue is less than the length of the minQueue
            # then assign the current index to the minQueue to find the shortest queue
            if len(self.queue[queue]) < len(self.queue[minQueue]):
                minQueue = queue
        return minQueue


class MMN_queue(MMN):
    """Sub class of MMN that implements the queueing system.
    Either MM1 or MMN. MM1 is a special case of MMN where n=1.
    It can also be used to implement the supermarket queue with specifying {sample_size}.
    """

    # Implementation of supermarket queue.
    # ? You have to specify the number of queues that you want to use and then the scheduler will pick the shortest queue
    # ? from {sample_size} of queues
    def __init__(
        self,
        lambd_time_waiting: float,
        mu_time_leaving: float,
        number_of_queues: int,
        sample_size: int = -1,
        k_shape_parameter: int = -1,
    ):
        super().__init__(
            lambd=lambd_time_waiting, mu=mu_time_leaving, n=number_of_queues
        )
        self.random_arrival_gen: float

        if k_shape_parameter > 0:
            self.random_arrival_gen = weibull_generator(
                k_shape_parameter, 1 / (number_of_queues * lambd_time_waiting)
            )
            self.random_completion_gen = weibull_generator(
                k_shape_parameter, 1 / (mu_time_leaving)
            )
        else:
            self.random_arrival_gen = partial(expovariate, self.lambd * self.n)
            self.random_completion_gen = partial(expovariate, self.mu)

        # ? Is the {n} amount of servers that we can use?
        if sample_size == -1 and number_of_queues > 1:
            self.supermarket_scheduler = False
            # pass
        else:
            if (
                sample_size > number_of_queues
                or sample_size < 1
                or number_of_queues < 1
            ):
                raise ValueError(
                    "Sample size cannot be greater than the number of queues that is not less than 1"
                )
            self.supermarket_scheduler = True
        # ? This is the number of queues that we will use for the scheduler
        self.sample_size: int = sample_size
        # ? This is the dictionary that will store the queue length distribution
        self.queue_len_distri: dict = {}

        # ? Why this?
        self.schedule_arrival(0)
        self.schedule(expovariate(lambd=self.lambd), Arrival(0))
        print(f"MMN_queue _> Created MMN_queue with {number_of_queues} queues")
        # ? Why this?
        # self.schedule(0, Log(1))

    def schedule_arrival(self, job_id):
        # Calling the generator regardless of its function
        self.schedule(self.random_arrival_gen(), Arrival(job_id))

    def schedule_completion(self, job_id):
        # Calling the generator regardless of its function
        self.schedule(self.random_completion_gen(), Completion(job_id))

    def retr_id_from_running(self, job_id: int):
        # get the index of the running job
        for i in range(len(self.running)):
            if self.running[i] == job_id:
                return i
        return None


class MM1(MMN):
    def __init__(self, lambd: float, mu: float):
        super().__init__(lambd=lambd, mu=mu, n=1)


class Arrival(Event):
    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: MMN | MMN_queue):
        # set the arrival time of the job
        arrival_time: float = sim.time_of_simulation
        sim.arrivals[self.id] = arrival_time
        # Supermarket queue:
        # for sample d in sim.n I will pick a queue that has the shortest length
        if sim.supermarket_scheduler:
            target_queue = sim.supermarket_choice()
        else:
            target_queue = choice(range(0, sim.n))
        # if there is no running job, assign the incoming one and schedule its completion
        if sim.running[target_queue] is None:
            sim.running[target_queue] = self.id
            sim.schedule_completion(self.id)
            # sim.schedule_completion(sim.running[target_queue])
        # otherwise put the job into the queue
        else:
            sim.queue[target_queue].append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)
        # print(f"Arrival _> Job {self.id} arrived at {arrival_time}")


class Completion(Event):
    # currently unused, might be useful when extending
    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: MMN | MMN_queue):
        # assert if the job is not currently running
        assert self.id in sim.running

        # * This is the index inside of the list of running jobs for the current running job
        current_job_queue_idx: int = sim.retr_id_from_running(job_id=self.id)
        # set the completion time of the running job
        sim.completions[
            sim.running[current_job_queue_idx]
        ] = sim.time_of_simulation
        # if the queue is not empty
        if len(sim.queue[current_job_queue_idx]) > 0:
            # get a job from the queue
            # ? This cannot be sim.running because of priority of timming
            _temp_job_id: int = sim.queue[current_job_queue_idx].popleft()
            # Putn the job into execution
            sim.running[current_job_queue_idx] = _temp_job_id
            # schedule its completion
            sim.schedule_completion(_temp_job_id)
        else:
            # otherwise set the running job to None
            sim.running[current_job_queue_idx] = None
        # print(f"Completion _> Job {self.id} completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambd", type=float, default=0.7)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--max-t", type=float, default=20_000)
    parser.add_argument("--n", type=int, default=10)

    parser.add_argument("--sample_size", type=int, default=-1)
    # Additional weilbull distribution parameter
    parser.add_argument("--weibull", type=int, default=None)

    parser.add_argument(
        "--csv", required=False, help="CSV file in  store results"
    )

    args = parser.parse_args()

    weibull_shape_parameter: int = args.weibull if args.weibull else -1

    simulation = MMN_queue(
        lambd_time_waiting=args.lambd,
        mu_time_leaving=args.mu,
        number_of_queues=args.n,
        k_shape_parameter=weibull_shape_parameter,
        sample_size=args.sample_size,
    )
    simulation.run(args.max_t)
    print(f"Main_>: finished simulation")
    completions = simulation.completions
    W = (
        sum(completions.values())
        - sum(simulation.arrivals[job_id] for job_id in completions)
    ) / len(completions)
    time_random_choice: float = 1 / (args.mu * (1 - (args.lambd / args.mu)))
    print(f"Average time spent in the system: {W}")
    print(
        f"Theoretical expectation for random server choice: {time_random_choice}"
    )
    # Old theoretical 1 / (1 - args.lambd)
    if args.csv is not None:
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == "__main__":
    main()

# Create a class for MMN queue
