"""Base classes for streaming analysis processes.

An *analysis process* is a small, long-lived program that subscribes to one or
more pub/sub streams, receives ``(message, head, data)`` messages in a loop,
transforms each one, and republishes the result on its own name. The existing
scripts in this package (``data_average.py``, ``imageselector.py``,
``subtract_bg.py``, ``rotate_image.py``) all hand-roll the same skeleton:

    * hold a :class:`Properties` connection named after the process,
    * declare the input streams as a :class:`PropertyAttribute`,
    * create a client, subscribe, then loop ``recv -> transform -> send``.

This module factors that skeleton into :class:`AnalysisProcess`, and captures
the only real differences between image and data analyses in two subclasses:

    * :class:`ImageAnalysis` -- uses an :class:`ImageClient`, subscribes to the
      ``imagestreams`` property, and republishes on channel ``'_'``.
    * :class:`DataAnalysis` -- uses a :class:`DataClient`, subscribes to the
      ``datastreams`` property, and republishes on the default channel.

Both clients publish with the same ``client.send(header, data, channel=...)``
order, so :meth:`AnalysisProcess.send_result` is shared by both subclasses.

To write a new analysis, subclass one of the two and override :meth:`process`::

    class ImageSlice(ImageAnalysis):
        _index = PropertyAttribute('SelectedImage', 1)

        def process(self, head, data):
            if head['_imgindex'] == self._index:
                return head, data      # publish this one
            return None                # drop everything else

    if __name__ == '__main__':
        run_analysis(ImageSlice)
"""

import argparse

from pytweezer.servers import DataClient, ImageClient
from pytweezer.servers import Properties, PropertyAttribute


class AnalysisProcess:
    """Common skeleton for a streaming analysis process.

    Subclasses supply the transport-specific pieces via the three hooks
    :meth:`make_client`, :meth:`send_result` and the ``_streams`` property, and
    the analysis-specific behaviour by overriding :meth:`process`.

    Attributes set up by ``__init__``:
        name    -- the process name (also the Properties namespace / title).
        _props  -- the :class:`Properties` connection.
        client  -- the pub/sub client (an ``ImageClient`` or ``DataClient``).
    """

    #: The input streams. Subclasses override this with the property name that
    #: matches their transport (``imagestreams`` / ``datastreams``).
    _streams = PropertyAttribute('streams', ['None'])

    #: Sub-channel appended to ``name`` when republishing results.
    _out_channel = ''

    def __init__(self, name):
        self.name = name
        self._props = Properties(name)

        self.client = self.make_client(name.split('/')[-1])
        self.client.subscribe(self._streams)
        print('{} ({}) subscriptions: {}'.format(
            type(self).__name__, name, self._streams))

    # ---- transport-specific hooks (implemented by ImageAnalysis/DataAnalysis)

    def make_client(self, name):
        """Return the pub/sub client for this analysis (subscribing publisher)."""
        raise NotImplementedError

    def send_result(self, head, data):
        """Publish one transformed ``(head, data)`` result on ``self.client``."""
        self.client.send(head, data, channel=self._out_channel)

    # ---- analysis-specific hook (implemented by concrete scripts) ----------

    def process(self, head, data):
        """Transform one received message.

        Args:
            head: the message header dict.
            data: the payload (numpy array for images/data).

        Returns:
            ``(head, data)`` to republish, or ``None`` to drop the message.
            The default is a pass-through.
        """
        return head, data

    # ---- main loop ---------------------------------------------------------

    def run(self):
        while True:
            msg = self.client.recv()
            if msg is None:
                continue
            msgstr, head, data = msg

            result = self.process(head, data)
            if result is not None:
                out_head, out_data = result
                self.send_result(out_head, out_data)


class ImageAnalysis(AnalysisProcess):
    """Analysis process that consumes and republishes an image stream."""

    _streams = PropertyAttribute('imagestreams', ['None'])
    _out_channel = '_'

    def make_client(self, name):
        return ImageClient(name)


class DataAnalysis(AnalysisProcess):
    """Analysis process that consumes and republishes a data stream."""

    _streams = PropertyAttribute('datastreams', ['None'])
    _out_channel = ''

    def make_client(self, name):
        return DataClient(name)


def run_analysis(analysis_cls, argv=None):
    """Standard ``main`` for an analysis script.

    Parses the single ``name`` argument the process launcher passes and runs an
    instance of ``analysis_cls``::

        if __name__ == '__main__':
            run_analysis(MyAnalysis)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args(argv)
    name = args.name[0]
    analysis_cls(name).run()


if __name__ == '__main__':
    pass
