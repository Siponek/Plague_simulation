from abc import abstractmethod
import logging
import heapq

# TODO: implement the event queue!
# suggestion: have a look at the heapq library (https://docs.python.org/dev/library/heapq.html)
# and in particular heappush and heappop


class Event:
    """
    Subclass this to represent your events.

    You may need to define __init__ to set up all the necessary information.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def process(self, sim):
        """
        Process the event.

        This is called by the simulation when the event is processed.
        """
        pass

    def __lt__(self, other):
        return id(self) < id(other)


class Simulation:
    """Subclass this to represent the simulation state.

    Here, self.t is the simulated time and self.events is the event queue.
    """

    def __init__(self) -> None:
        """Extend this method with the needed initialization.

        You can call super().__init__() there to call the code here.
        """

        # simulated time
        self.time_of_simulation: int = 0
        # TODO: set up self.events as an empty queue
        self.event_queue: list = []
        heapq.heapify(self.event_queue)

    def schedule(self, delay: float, event: Event) -> None:
        """Add an event to the event queue after the required delay."""

        # TODO: add event to the queue at time self.time_of_simulation + delay
        heapq.heappush(
            self.event_queue, (self.time_of_simulation + delay, event)
        )
        return None

    def run(self, max_t=float("inf")) -> None:
        """Run the simulation. If max_t is specified, stop it at that time."""

        # TODO: as long as the event queue is not empty:
        while len(self.event_queue) > 0:
            # TODO: get the first event from the queue
            time_of_simulation, event = heapq.heappop(self.event_queue)
            if time_of_simulation > max_t:
                break
            self.time_of_simulation = time_of_simulation
            event.process(self)

    def log_info(self, msg):
        logging.info(f"{self.time_of_simulation:.2f}: {msg}")
