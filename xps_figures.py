
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from collections import ChainMap


class MultipleFigures(object):

    def __init__(self, data, settings, data_type='avantage_export'):
        """Init local variables

        Args:
            data (object): The data to be plotted
            settings (dict): Specify settings

        """
        self.gridspec = gridspec.GridSpec(*settings['gridspec'])

        if data_type == 'avantage_export':
            self._plot_from_avantage_export(data, settings)
        else:
            raise ValueError('Unkown data type: {}'.format(data_type))

        self._finalize_figure(settings)

    @classmethod
    def from_avantage_export(cls, data, layout):
        return cls.__init__(export, layout, format='avantage_export')

    def _plot_from_avantage_export(self, data, settings):
        """plot from Avantage export"""
        graphs = settings['graphs']
        for graph in graphs:
            # Chained settings and graph settings
            chained_settings = ChainMap(graph, settings)

            # Get the data
            graph_data = data[graph['key']]

            # Get supplot and graph specific settings
            axes = plt.subplot(self.gridspec[graph['grid_pos']])

            # Extract x and plot counts
            x = graph_data['Binding Energy (E)']
            linewidth = graph.get('linewidth', 2)
            axes.plot(x, graph_data['Counts'], label='Counts',
                      linewidth=linewidth)

            # Check if there is a background and plot that
            background = graph_data.get('Backgnd.')
            if background is not None:
                backgnd_label = graph.get('backgnd_label', graph['key'].split(' ')[0] + ' backgnd.')
                mask = np.where(background > 0.1)
                axes.plot(x[mask], background[mask], linewidth=1, label=backgnd_label)

            # Plot fits
            if graph_data.get('Envelope') is not None:
                graph_data_copy = graph_data.copy()
                envelope = graph_data_copy.pop('Envelope')
                for exclude_key in ('Binding Energy (E)', 'Backgnd.', 'Residuals', 'Counts'):
                    try:
                        graph_data_copy.pop(exclude_key)
                    except KeyError:
                        pass
                axes.plot(x, envelope, label='Envelope')

                for fit_name, fit_data in graph_data_copy.items():
                    mask = np.where(fit_data > 0.1)
                    axes.plot(x[mask], fit_data[mask], color='#AEB404')

            self._format_axes(axes, chained_settings)

    def _format_axes(self, axes, graph):
        """Apply different"""
        loc = graph.get('loc', 'best')
        legend_fontsize = graph.get('legend_fontsize', 'medium')
        axes.legend(loc=loc, fontsize=legend_fontsize)

        # Adjust xlim and reverse x-axes
        x_limits = list(axes.get_xlim())
        for n, limit in enumerate(graph.get('xlim', ())):
            if limit is not None:
                x_limits[n] = limit
        axes.set_xlim(reversed(x_limits))

        # Adjust ylim
        y_limits = list(axes.get_ylim())
        if graph.get('ylim') is not None:
            for n, limit in enumerate(graph['ylim']):
                if limit is not None:
                    y_limits[n] = limit
            axes.set_ylim(y_limits)

        # Apply title
        title = graph.get('title', graph['key'].split(' ')[0])
        axes.set_title(title)

    def _finalize_figure(self, settings):
        """Finalize figure"""
        plt.tight_layout()

    def show(self):
        plt.show()


def main():
    from PyExpLabSys.file_parsers.avantage_xlsx_export import AvantageXLSXExport
    avexport = AvantageXLSXExport("/home/kenni/Dokumenter/xps/soren_dahl_battery/soren_dahl_battery/HT1/peak table.xlsx")

    graphs = [
        {'key': 'Ni2p Scan', 'grid_pos': (0, 0), 'loc': 'lower left', 'backgnd_label': 'Ni2p3 backgnd.'},
        {'key': 'Mn2p Scan', 'grid_pos': (0, 1), 'backgnd_label': 'Mn2p3 backgnd.'},
        {'key': 'S2p Scan more scans', 'grid_pos': (1, 0), 'xlim': (160, None), 'ylim': (320, 400)},
        {'key': 'O1s Scan', 'grid_pos': (1, 1)},
        {'key': 'C1s Scan', 'grid_pos': (2, 0)},
        {'key': 'Si2p Scan', 'grid_pos': (2, 1)},
        
    ]
    settings = {
        'legend_fontsize': 'small',
        'gridspec': (3, 2),
        'graphs': graphs,
    }
    mulfig = MultipleFigures(avexport, settings)
    mulfig.show()


if __name__ == '__main__':
    main()
