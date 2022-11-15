##!/usr/bin/env python

import argparse
import csv
import collections
from random import expovariate, choice

from discrete_event_sim import Simulation, Event

# ? can we use random.expovariate() here?
# from weibull import weibull_generator

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
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.arrival_rate = lambd / n
        self.completion_rate = mu / n
        self.schedule(expovariate(lambd=lambd), Arrival(0))

    def schedule_arrival(self, job_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        # Rate of arrival will be lambda*n
        self.schedule(expovariate(self.lambd * self.n), Arrival(job_id))

    def schedule_completion(self, job_id):
        # schedule the time of the completion event
        self.schedule(expovariate(lambd=self.mu), Completion(job_id))

    @property
    def queue_len(self):
        # todo: this is not correct, it should be the sum of the queue lengths
        # todo it needs to return list of length of queues
        return [True for x in self.running if x is None] + [
            len(x) for x in self.queue
        ]
        # return (self.running[0] is None) + len(self.queue)


class MMN_queue(MMN):
    def __init__(self, lambd: float, mu: float, n: int):
        super().__init__(lambd, mu, n)


class MM1(MMN):
    def __init__(self, lambd, mu):
        super().__init__(lambd, mu, 1)


class Arrival(Event):
    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: MMN):
        # set the arrival time of the job
        arrival_time: float = sim.time_of_simulation
        sim.arrivals[self.id] = arrival_time
        # if there is no running job, assign the incoming one and schedule its completion
        temp_queue = choice(range(0, sim.n))
        if sim.running[temp_queue] is None:
            sim.running[temp_queue] = self.id
            sim.schedule_completion(sim.running[temp_queue])
        # otherwise put the job into the queue
        else:
            sim.queue[temp_queue].append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)


class Completion(Event):
    # currently unused, might be useful when extending
    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: MMN):
        # assert if the job is not currently running
        # print("Completion of job: ", self.id)
        # print("Running jobs: ", sim.running)
        assert self.id in sim.running
        # set the completion time of the running job

        # * This is the index inside of the list of running jobs for the current running job
        temp_idx: int = self.retr_id_from_running(sim)
        sim.completions[sim.running[temp_idx]] = sim.time_of_simulation
        # if the queue is not empty
        if len(sim.queue[temp_idx]) > 0:
            # get a job from the queue
            # ? This cannot be sim.running because of priority of timming
            _temp_job_id: int = sim.queue[temp_idx].popleft()
            # Putn the job into execution
            sim.running[temp_idx] = _temp_job_id
            # schedule its completion
            sim.schedule_completion(_temp_job_id)
        else:
            sim.running[temp_idx] = None

    def retr_id_from_running(self, sim: MMN):
        # get the index of the running job
        for i in range(len(sim.running)):
            if sim.running[i] == self.id:
                return i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambd", type=float, default=0.7)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--max-t", type=float, default=1_000_000)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument(
        "--csv", required=False, help="CSV file in  store results"
    )
    args = parser.parse_args()

    sim = MMN(args.lambd, args.mu, args.n)
    sim.run(args.max_t)

    completions = sim.completions
    W = (
        sum(completions.values())
        - sum(sim.arrivals[job_id] for job_id in completions)
    ) / len(completions)
    print(f"Average time spent in the system: {W}")
    print(
        f"Theoretical expectation for random server choice: {1 / (1 - args.lambd)}"
    )

    if args.csv is not None:
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == "__main__":
    main()

# Create a class for MMN queue
