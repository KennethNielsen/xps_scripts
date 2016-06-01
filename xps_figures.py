
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


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
            axes = plt.subplot(self.gridspec[graph['grid_pos']])
            graph_data = data[graph['key']]
            linewidth = graph.get('linewidth', 2)
            axes.plot(graph_data['Binding Energy (E)'], graph_data['Counts'], linewidth=linewidth)
            background = graph_data.get('Backgnd.')
            if background is not None:
                axes.plot(graph_data['Binding Energy (E)'], background, linewidth=1)
            
            self._format_axes(axes, graph)

    def _format_axes(self, axes, graph):
        """Apply different"""
        # Reverse x-axes
        axes.set_xlim(reversed(axes.get_xlim()))

        # Apply title
        axes.set_title(graph['title'])

    def _finalize_figure(self, settings):
        """Finalize figure"""
        plt.tight_layout()

    def show(self):
        plt.show()
            



def main():
    from PyExpLabSys.file_parsers.avantage_xlsx_export import AvantageXLSXExport
    avexport = AvantageXLSXExport("/home/kenni/Dokumenter/xps/soren_dahl_battery/soren_dahl_battery/HT1/peak table.xlsx")

    graphs = [
        {'key': 'Ni2p Scan', 'grid_pos': (0, 0), 'title': 'Ni2p'},
        {'key': 'Mn2p Scan', 'grid_pos': (0, 1), 'title': 'Mn2p'},
        {'key': 'S2p Scan more scans', 'grid_pos': (1, 0), 'title': 'S2p'},
        {'key': 'O1s Scan', 'grid_pos': (1, 1), 'title': 'O1s'},
        {'key': 'C1s Scan', 'grid_pos': (2, 0), 'title': 'C1s'},
        {'key': 'Si2p Scan', 'grid_pos': (2, 1), 'title': 'Si2p'},
        
    ]
    settings = {
        'gridspec': (3, 2),
        'graphs': graphs,
    }
    mulfig = MultipleFigures(avexport, settings)
    mulfig.show()


if __name__ == '__main__':
    main()
