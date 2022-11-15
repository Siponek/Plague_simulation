#!/usr/bin/env python

import argparse
import collections
import csv
from random import expovariate, randrange, sample
from discrete_event_sim_redacted import Simulation, Event
from matplotlib import pyplot as plt


# To use weibull variates, for a given set of parameter do something like
# from weibull import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


class MMN(Simulation):

    def __init__(self, lambd, mu, n):
        #if n != 1:
        #    raise NotImplementedError  # extend this to make it work for multiple queues

        super().__init__()
        self.running = [None for _ in range(n)]  # if not None, the id of the running job

        self.queue = [collections.deque() for _ in range(n)]  # FIFO queue of the system

        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.arrival_rate = lambd / n
        self.completion_rate = mu / n
        self.schedule_arrival(0)
        self.schedule(0, Log(10))   # logging initial queue situation

    def schedule_arrival(self, job_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        job_t = expovariate(self.lambd * self.n)
        #print(f"Scheduled arrival of job {job_id} at {self.t + job_t}")
        self.schedule(job_t, Arrival(job_id))

    def schedule_completion(self, job_id):
        # schedule the time of the completion event
        job_t = expovariate(self.mu)
        #print(f"Scheduled completion of job {job_id} at {self.t + job_t}")
        self.schedule(job_t, Completion(job_id))

    @property
    def queue_len(self):
        return (self.running[0] is None) + len(self.queue[0])


class MMNImpl(MMN):

    def __init__(self, lambd, mu, n, d):

        if d == -1 and n > 0:
            self.supermarket_mode = False
        else:
            if d > n or d <= 0 or n <= 0:  # sample dimension can't be negative neither greater than the number of queues
                raise RuntimeError
            self.supermarket_mode = True

        super().__init__(lambd, mu, n)
        self.sample_dim = d
        self.queue_len_distr = {}


    # return the index of the queue belonging to the sample
    # having the actual minimum length
    def supermarket_decision(self):
        mySample = sample([i for i in range(self.n)], self.sample_dim)
        if len(mySample) == 1:
            return mySample[0]

        minQueue = mySample[0]

        for qi in mySample:
            if len(self.queue[qi]) < len(self.queue[minQueue]):
                minQueue = qi
        return minQueue

    def get_job_executor(self, job_id):
        for i in range(len(self.running)):
            if self.running[i] == job_id:
                return i
        return None

    def register_queue_lengths(self, t):
        print(f"{t} -> Registering queue lengths...")
        self.queue_len_distr[t] = self.queues_len

    @property
    def queues_len(self):
        return [len(queue) for queue in self.queue]


class Arrival(Event):

    def __init__(self, job_id):
        self.id = job_id  # assigning the job id

    def process(self, sim: MMNImpl):
        #print(f"Arrivato job {self.id}")
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        # if there is no running job, assign the incoming one and schedule its completion
        if not sim.supermarket_mode:
            insert_target = randrange(sim.n)
        else:
            insert_target = sim.supermarket_decision()

        if sim.running[insert_target] is None:  # look if there is a server that is free, then assign the job to him
            #print(f"Eseguendo job {self.id}...")
            sim.running[insert_target] = self.id
            sim.schedule_completion(self.id)
        else:
            # deciding which queue the job will be inserted into
            sim.queue[insert_target].append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)


class Completion(Event):

    def __init__(self, job_id):
        self.id = job_id  # currently unused, might be useful when extending

    def process(self, sim: MMNImpl):
        job_executor = sim.get_job_executor(self.id)
        assert job_executor is not None
        #print(f"Completato job {self.id}")
        # set the completion time of the running job
        sim.completions[self.id] = sim.t
        if len(sim.queue[job_executor]) > 0:
            # get a job from the queue
            pop_target = job_executor   # deciding from which queue we'll extract the next job
            job_id = sim.queue[pop_target].popleft()  # popping a task from the start of the queue
            sim.running[pop_target] = job_id
            # schedule its completion
            sim.schedule_completion(job_id)
        else:
            sim.running[job_executor] = None


class Log(Event):

    def __init__(self, period):
        self.period = period

    def process(self, sim: MMNImpl):
        sim.register_queue_lengths(sim.t)
        sim.schedule(self.period, Log(self.period))


def fraction_of_queues_of_length(min_l, queue_distr, n_queues):
    counter = 0

    for k, v in queue_distr.items():
        for q_len in v:
            if q_len >= min_l:
                counter = counter + 1

    return counter / (len(queue_distr.keys()) * n_queues)


def get_all_qlengths(queue_distr):
    points = []

    for k, v in queue_distr.items():
        for q_len in v:
            if q_len not in points:
                points.append(q_len)
    return sorted(points)


def plot_supermarket_graphs(queue_distr, n_queues):
    points = get_all_qlengths(queue_distr)
    percs = [fraction_of_queues_of_length(min_l, queue_distr, n_queues) for min_l in points]

    plt.plot(points, percs, label="Supermarket model")
    plt.xlabel("Queue length")
    plt.ylabel("Fraction of queues with at least that size")
    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=0.95)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1000000)
    parser.add_argument('--n', type=int, default=10)

    # needed to set parameter d (sample dimension for supermarket model choice of the queue)
    parser.add_argument('--d', type=int, default=-1)

    parser.add_argument('--csv', help="sim.arrivals[self.id] = sim.t", default="report.csv")
    args = parser.parse_args()

    sim = MMNImpl(args.lambd, args.mu, args.n, args.d)

    sim.run(args.max_t)

    completions = sim.completions

    if sim.supermarket_mode:
        plot_supermarket_graphs(sim.queue_len_distr, args.n)
    else:
        val = 1 / (1 - sim.lambd)
        print(f"Theoretical expectation for random server choice: {val}")
    W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
    print(f"Average time spent in the system: {W}")

    if args.csv is not None:
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W, args.d, args.n])


if __name__ == '__main__':
    main()
