#!/usr/bin/env python

import argparse
import cmath
import csv
import math
import os
from random import expovariate, randrange, sample, seed, weibullvariate
from discrete_event_sim_redacted import Simulation, Event
from matplotlib import pyplot as plt

# To use weibull variates, for a given set of parameter do something like
# from weibull import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable
from workloads import weibull_generator


class MMN(Simulation):

    def __init__(self, lambd, mu, n, shape):

        super().__init__()
        self.running = [None for _ in range(n)]  # if not None, the id of the running job

        self.queue = [[] for _ in range(n)]  # FIFO queue of the system

        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n  # number of queues
        self.mu = mu
        self.arrival_rate = lambd / n
        self.completion_rate = mu / n
        self.sample_dim = 1

        # if shape parameter is given, we'll generate delays using weibull distribution with given shape
        if shape > 0:
            self.arrival_gen = weibull_generator(shape, 1 / (n * lambd))  # mean has to be 1 / (n * lambda)
            self.completion_gen = weibull_generator(shape, 1 / mu)  # mean has to be 1 / mu
        else:  # we'll generate delays using exponential distribution
            self.arrival_gen = None
            self.completion_gen = None

    def schedule_arrival(self, job_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        job_t = self.arrival_gen() if self.arrival_gen is not None else expovariate(self.lambd * self.n)
        self.schedule(job_t, Arrival(job_id))

    def schedule_completion(self, job_id):
        # schedule the time of the completion event
        job_t = self.completion_gen() if self.completion_gen is not None else expovariate(self.mu)
        self.schedule(job_t, Completion(job_id))

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

    @property
    def queue_len(self):
        return [len(list(q)) for q in self.queue]


class MMNImpl(MMN):

    def __init__(self, lambd, mu, n, d, shape):

        # not using supermarket scheduling
        if d == -1 and n > 0:
            self.supermarket_mode = False
        else:
            # sample dimension can't be negative neither greater than the number of queues
            if d > n or d <= 0 or n <= 0:
                raise RuntimeError
            self.supermarket_mode = True

        super().__init__(lambd, mu, n, shape)
        self.sample_dim = d  # sample dimension for supermarket scheduling
        self.queue_len_distr = {}  # mapping time to list of queues' lengths
        self.schedule_arrival(0)
        self.schedule(0, Log(1))  # logging initial queue situation

    # returns the index of the server that is executing job_id if it is found, None otherwise
    def get_job_executor(self, job_id):
        for i in range(len(self.running)):
            if self.running[i] == job_id:
                return i
        return None

    # registering queue lengths
    def register_queue_lengths(self, t):
        self.queue_len_distr[t] = self.queues_len

    # property representing the list of queues' lengths
    @property
    def queues_len(self):
        return [len(queue) for queue in self.queue]

    # runs the simulation
    def simulate(self, max_t, shape, file, plot_file):
        self.run(max_t)

        # retrieving average time spent by a job in the system
        completions = self.completions
        W = (sum(completions.values()) - sum(self.arrivals[job_id]
                                             for job_id in completions)) / len(completions)

        # plotting supermarket queue lengths over time
        if self.supermarket_mode:
            plot_supermarket_graphs(self.queue_len_distr, self.n, self.sample_dim,
                                    self.lambd, plot_file, self.mu)
        print(f"Average time spent in the system by a job: {W}")
        val = 1 / (self.mu * (1 - (self.lambd / self.mu)))  # theoretical expectation for random choice
        print(f"Theoretical expectation for random server choice: {val}")

        if file is not None:
            with open(file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.lambd, self.mu, max_t, W, self.sample_dim, self.n, shape])


# simulation for round robin scheduling
class MMNRoundRobinSim:

    def __init__(self, lambd, mu, n, d, max_t, shape, file, plot_file):

        # change first parameter in order to vary the shape of weibull distribution
        mean_cd = 1 / mu
        mean_ad = 1 / (n * lambd)
        generator_cd = weibull_generator(shape, mean_cd) if shape > 0 else None
        generator_ad = weibull_generator(shape, mean_ad) if shape > 0 else None
        self.quantums = []  # quantums to be tested

        # istances of round robin simulation, mapped through assigned quantum of time
        self.systems = {}
        # mapping job_id - delay for arrival
        self.arrivals_delay = {}

        # initializing delays for jobs in order to apply the same delays to all the instances
        init_t = 0
        ji = 0
        while init_t < max_t:
            # generate delays using weibull distribution if shape is provided, otherwise uses exponential
            self.arrivals_delay[ji] = generator_ad() if generator_ad is not None else expovariate(n * lambd)
            init_t += self.arrivals_delay[ji]
            ji = ji + 1

        # mapping job_id - execution time
        self.completion_delay = {}
        for job_id in list(self.arrivals_delay.keys()):
            self.completion_delay[job_id] = generator_cd() if generator_cd is not None else expovariate(mu)

        # retrieving maximum execution time so when the simulation will run with it as the quantum
        # of time, the behaviour will be the same as FIFO queue
        max_q = max(self.completion_delay.values())
        min_q = min(self.completion_delay.values())
        self.quantums.append(max_q + 1.0)
        self.quantums.append(min_q + 1.0)

        tmp = mu

        # generating quantums with decresing power of 2
        # until the quantum is about 1/100 of mu
        while tmp > mu * pow(10, -2):
            self.quantums.append(tmp)
            tmp = tmp / 2

        self.quantums = sorted(self.quantums)  # order the quantums

        quantum_times = {}

        # multiple runs of round robin scheduling with the generated quantums
        for quantum in self.quantums:
            self.systems[quantum] = MMNRoundRobin(lambd, mu, n, quantum, d, self.arrivals_delay, self.completion_delay)
            self.systems[quantum].run(max_t)
            completions = self.systems[quantum].completions
            # calculating average time spent in the system by a job
            W = (sum(completions.values()) - sum(self.systems[quantum].arrivals[job_id]
                                                 for job_id in completions)) / len(completions)
            print(f"{quantum} => Average time spent in the system by a job: {W}")
            quantum_times[quantum] = W

        # retrieving simulation parameters to show
        info_dict = {"lambda": lambd,
                     "mu": mu,
                     "min_q": min_q,
                     "max_q": max_q - 1.0,
                     "shape": shape,
                     "n": n,
                     "d": d,
                     "plot_file": plot_file}

        # plotting
        self.plot_time_quantum(quantum_times, info_dict)

        # saving to disk
        if file is not None:
            with open(file, 'a', newline='') as f:
                writer = csv.writer(f)

                for system in self.systems.values():
                    writer.writerow([lambd, mu, max_t, system.quantum, d, n,
                                     quantum_times[system.quantum], shape])

    # plot relationship between time quantum and average time spent by a job in the system
    def plot_time_quantum(self, quantum_times, info_dict):
        quantums = quantum_times.keys()
        avg_times = quantum_times.values()
        plt.plot(quantums, avg_times)
        plt.axvline(x=info_dict["min_q"], color='red', linestyle='--', label='min job duration')
        plt.axvline(x=info_dict["max_q"], color='green', linestyle='--', label='max job duration')

        # if weibull shape is not given, we use exponential distribution
        if info_dict["shape"] == -1:
            info_dict["shape"] = 1  # that is a special case of Weibull with shape = 1

        text = "lambda = " + str(info_dict["lambda"]) + "\nmu = " + str(info_dict["mu"]) + \
               "\nnumber of servers = " + str(info_dict["n"]) + "\nsample dimension = " + \
               str(info_dict["d"]) + "\nweibull shape = " + str(info_dict["shape"])

        # textbox properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gcf().text(0.70, 0.85, text, fontsize=8, bbox=props)

        plt.title("Round Robin scheduling")
        plt.xlabel("Quantum of time")
        plt.ylabel("Average time spent in the system")
        plt.legend(loc=0)
        plt.grid()

        if info_dict["plot_file"] is not None:
            plt.savefig(info_dict["plot_file"])


class MMNRoundRobin(MMN):

    def __init__(self, lambd, mu, n, quantum, d, ad, cd):
        # distinguish if we are using supermarket decision or not
        if d == -1 and n > 0:
            self.supermarket_mode = False
        else:
            # sample dimension can't be negative neither greater than the number of queues
            if d > n or d <= 0 or n <= 0:
                raise RuntimeError
            self.supermarket_mode = True
        self.sample_dim = d
        super().__init__(lambd, mu, n, -1)

        # a running job is represented by the tuple (job_id, remaining execution time)
        self.running = [(None, None) for _ in range(self.n)]
        self.quantum = quantum  # quantum of time to assign to each job
        self.ad = ad  # arrival delays for each job
        self.cd = cd  # execution time for each job
        self.schedule_arrival(0)

    # returns the index of the server that is running a job, None if it is not found
    def get_job_executor_rr(self, job_id):
        for i in range(len(self.running)):
            j, t = self.running[i]
            if j == job_id:
                return i
        return None

    # schedule a completion or an interruption event basing on remaining execution time
    def schedule_completion_rr(self, job_id, rem_time):
        if rem_time > self.quantum:  # job needs to be interrupted at least once
            self.schedule(self.quantum, CompletionRR(job_id, True, rem_time))
        else:  # job can be executed till the end with no interruption
            self.schedule(rem_time, CompletionRR(job_id, False, rem_time))

    # schedules arrival of a job
    def schedule_arrival(self, job_id):
        job_t = self.ad[job_id]  # delay for arrival of job_id
        self.schedule(job_t, ArrivalRR(job_id))


class ArrivalRR(Event):

    def __init__(self, job_id):
        self.id = job_id  # assigning the job id

    def process(self, sim: MMNRoundRobin):
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t

        # deciding which queue will be checked
        if sim.supermarket_mode:
            queue_to_ins = sim.supermarket_decision()  # the least loaded one
        else:
            queue_to_ins = randrange(sim.n)  # a random one

        # server associated with selected queue
        j, t = sim.running[queue_to_ins]

        if j is None and t is None:  # if the server is free, put the job into execution
            exec_time = sim.cd[self.id]  # get execution time for the specific job
            sim.running[queue_to_ins] = (self.id, exec_time)  # tuple (job_id, time the system will need to serve it)
            sim.schedule_completion_rr(self.id, exec_time)  # schedule its completion (or interruption)
        else:
            # a job when first arrives is inserted in the queue
            sim.queue[queue_to_ins].append((self.id, sim.cd[self.id]))
        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)


class CompletionRR(Event):

    def __init__(self, job_id, interruption, rem_time):
        self.id = job_id  # assigning the job id
        self.interruption = interruption  # flag to indicate if the completion event is an interruption
        self.rem_time = rem_time  # remaining execution time of the job

    def process(self, sim: MMNRoundRobin):
        job_executor = sim.get_job_executor_rr(self.id)  # first find server that the job is running on
        assert job_executor is not None  # if it is None, then it's an error
        job_id, rem_time = sim.running[job_executor]  # retrieve job and remaining execution time

        if self.interruption:  # job stopped due to interruption
            # update remaining executon time
            new_rem_time = rem_time - sim.quantum
            # take the job from running and insert it back in the queue
            sim.queue[job_executor].append((job_id, new_rem_time))
            # pop job from the queue and execute it
            job, exec_t = sim.queue[job_executor].pop(0)

            sim.running[job_executor] = (job, exec_t)  # job runs for the remaining time
            sim.schedule_completion_rr(job, exec_t)  # schedule quantum interruption (or completion) for the new job
        else:  # job finished before quantum of time assigned
            # set the completion time of the running job
            sim.completions[self.id] = sim.t

            if len(sim.queue[job_executor]) > 0:  # server has other jobs waiting to be executed
                # get a job from the queue
                pop_target = job_executor  # deciding from which queue we'll extract the next job
                job_id, exec_t = sim.queue[pop_target].pop(0)  # popping a task from the start of the queue
                sim.running[job_executor] = (job_id, exec_t)  # job runs for the remaining time
                # schedule quantum interruption (or completion) for the new job
                sim.schedule_completion_rr(job_id, exec_t)
            else:
                sim.running[job_executor] = (None, None)  # server is free


class Arrival(Event):

    def __init__(self, job_id):
        self.id = job_id  # assigning the job id

    def process(self, sim: MMNImpl):
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        # if there is no running job, assign the incoming one and schedule its completion
        if not sim.supermarket_mode:
            insert_target = randrange(sim.n)
        else:
            insert_target = sim.supermarket_decision()

        # look if there is a server that is free, then assign the job to him
        if sim.running[insert_target] is None:
            sim.running[insert_target] = self.id
            sim.schedule_completion(self.id)  # schedule job completion
        else:
            # insert the job in the queue
            sim.queue[insert_target].append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)


class Completion(Event):

    def __init__(self, job_id):
        self.id = job_id  # currently unused, might be useful when extending

    def process(self, sim: MMNImpl):
        job_executor = sim.get_job_executor(self.id)  # first find server that the job is running on
        assert job_executor is not None  # if it is None, then it's an error
        # set the completion time of the running job
        sim.completions[self.id] = sim.t
        if len(sim.queue[job_executor]) > 0:  # server has other jobs waiting to be executed
            pop_target = job_executor  # deciding from which queue we'll extract the next job
            job_id = sim.queue[pop_target].pop(0)  # popping a task from the start of the queue
            sim.running[pop_target] = job_id  # put the task into execution
            # schedule its completion
            sim.schedule_completion(job_id)
        else:
            sim.running[job_executor] = None  # server is free


class Log(Event):

    def __init__(self, period):
        self.period = period

    def process(self, sim: MMNImpl):
        # registering queue lengths
        sim.register_queue_lengths(sim.t)
        # scheduling the event in order to be periodic (so same period as this one)
        sim.schedule(self.period, Log(self.period))


# counts the fraction of queues that have at least min_l size
def fraction_of_queues_of_length(min_l, queue_distr, n_queues):
    counter = 0

    for k, v in queue_distr.items():
        for q_len in v:
            if q_len >= min_l:
                counter = counter + 1

    return counter / (len(queue_distr.keys()) * n_queues)


# get the list of all lengths registered
def get_all_qlengths(queue_distr):
    points = []

    for k, v in queue_distr.items():
        for q_len in v:
            if q_len not in points:
                points.append(q_len)
    return sorted(points)


# plots the graph lengths - fraction of queues with at least that size
def plot_supermarket_graphs(queue_distr, n_queues, sample_dim, lambd, plot_file, mu):
    points = get_all_qlengths(queue_distr)
    percs = [fraction_of_queues_of_length(min_l, queue_distr, n_queues) for min_l in points]

    plt.plot(points, percs, label="Supermarket model")
    plt.xlabel("Queue length")
    plt.ylabel("Fraction of queues with at least that size")
    title = "Supermarket model"
    text = "lambda = " + str(lambd) + "\nmu = " + str(mu) + \
           "\nnumber of servers = " + str(n_queues) + "\nsample dimension = " + \
           str(sample_dim)

    # textbox properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.70, 0.85, text, fontsize=8, bbox=props)

    plt.title(title)
    plt.grid()

    if plot_file is not None:
        plt.savefig(plot_file)


# utility function to get next plot filename
def get_next_plot_name(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    count = 0
    while True:
        filename = dirname + "/plot" + str(count) + ".jpg"
        if not os.path.isfile(filename):
            f = open(filename, "x")
            f.close()
            return filename
        count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=0.7)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--max-t', type=float, default=100000)
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--n', type=int, default=10)

    # needed to set parameter d (sample dimension for supermarket model choice of the queue)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--csv', help="sim.arrivals[self.id] = sim.t", default="report.csv")
    parser.add_argument('--csv-rr', help="report for round robin", default="report_rr.csv")

    # introduce additional parameters for distinguish between using weibull or exponential distribution
    parser.add_argument('--weibull', type=float, default=1)
    # differentiating between round robin execution or supermarket
    parser.add_argument('--rr', type=int, default=1)
    parser.add_argument('--plots-dir', type=str, default="./plots")
    args = parser.parse_args()

    if args.seed is not None:
        seed(args.seed)

    weibull_shape = args.weibull if args.weibull is not None else -1

    if args.rr == 1:
        print("Starting new Round Robin simulation")
        my_simulation = MMNRoundRobinSim(args.lambd, args.mu, args.n, args.d, args.max_t, weibull_shape, args.csv_rr,
                                         get_next_plot_name(args.plots_dir + "/extended"))
        print("Round Robin simulation finished")
    else:
        print("Starting new Supermarket simulation")
        my_simualtion = MMNImpl(args.lambd, args.mu, args.n, args.d, weibull_shape)
        my_simualtion.simulate(args.max_t, weibull_shape, args.csv, get_next_plot_name(args.plots_dir + "/base"))
        print("Supermarket simulation finished")


if __name__ == '__main__':
    main()
