from abc import ABC, abstractmethod


class Page(ABC):
    """A base class for pages"""

    def __init__(self, app):
        self.app = app
        self.layouts = {"basic": self.create_basic_layout()}
        self.register_callbacks()

    @abstractmethod
    def create_basic_layout(self):
        """Creates basic layout of the page"""
        pass

    def register_callbacks(self):
        """Registers callbacks"""

        pass
