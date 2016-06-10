
from __future__ import division

import numpy as np
import matplotlib as mpl
# Make a litle extra room around the axis labels
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap


class BaseXPSFigure(object):
    """Base XPS Figure"""

    def _finalize_figure(self, settings):
        """Finalize figure"""
        plt.tight_layout()

    def show(self):
        plt.show()

    def savefig(self, *args, **kwargs):
        """Call plt savefig"""
        plt.savefig(*args, **kwargs)

    def _format_axes_common(self, axes, graph, height_fractions):
        """Common format axes code"""
        loc = graph.get('loc', 'upper right')
        legend_fontsize = graph.get('legend_fontsize',
                                    self._subplot_y_size // height_fractions['legend'])
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

        # Change size of tick labels
        tick_label_size = graph.get('tick_label_size', self._subplot_y_size // height_fractions['tick_label'])
        axes.tick_params(axis='both', which='major', labelsize=tick_label_size)


        # Apply title
        title = graph.get('title', graph['key'].split(' ')[0])
        title_size = graph.get('title_size', self._subplot_y_size // height_fractions['title'])
        axes.set_title(title, fontsize=title_size)



class MultipleFigures(BaseXPSFigure):

    def __init__(self, data, settings, data_type='avantage_export',
                 figure_args={'figsize': (8, 6), 'dpi': 100}):
        """Init local variables

        Args:
            data (object): The data to be plotted
            settings (dict): Specify settings

        """

        self._subplot_x_size = figure_args['figsize'][0] * figure_args['dpi'] // settings['gridspec'][1]
        self._subplot_y_size = figure_args['figsize'][1] * figure_args['dpi'] // settings['gridspec'][0]

        self.gridspec = gridspec.GridSpec(*settings['gridspec'])
        self.figure = plt.figure(**figure_args)

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
            fit_legend_set = False
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
                    if not fit_legend_set:
                        legend = 'Fits'
                        fit_legend_set = True
                    else:
                        legend = None
                    axes.plot(x[mask], fit_data[mask], color='#AEB404', label=legend)

            self._format_axes(axes, chained_settings)

    def _format_axes(self, axes, graph):
        """Apply different"""
        height_fractions = {'legend': 30, 'tick_label': 28, 'title': 18}
        height_fractions.update(graph.get('height_fractions', {}))
        self._format_axes_common(axes, graph, height_fractions)

        # Legends
        label_size = graph.get('height_fractions', {}).get('label_size', self._subplot_y_size // 22)
        grid_pos = graph['grid_pos']
        # Only set the labels on the outer plots
        if grid_pos[1] == 0:
            axes.set_ylabel('Counts [cps]', fontsize=label_size)
        if grid_pos[0] == self.gridspec.get_geometry()[0] - 1:
            axes.set_xlabel('Binding energy [eV]', fontsize=label_size)


class Survey(BaseXPSFigure):
    def __init__(self, data, settings, data_type='avantage_export',
                 figure_args={'figsize': (18, 11), 'dpi': 100}):
        """Init local variables

        Args:
            data (object): The data to be plotted
            settings (dict): Specify settings

        """
        self._subplot_x_size = figure_args['figsize'][0] * figure_args['dpi']
        self._subplot_y_size = figure_args['figsize'][1] * figure_args['dpi']

        self.figure = plt.figure(**figure_args)
        self.axes = self.figure.add_subplot(111)

        if data_type == 'avantage_export':
            self._plot_from_avantage_export(data, settings)
        else:
            raise ValueError('Unkown data type: {}'.format(data_type))

        self._finalize_figure(settings)

    def _plot_from_avantage_export(self, data, settings):
        """"""
        graph_data = data[settings.get('key', 'Survey Scan')]
        self.axes.plot(graph_data['Binding Energy (E)'], graph_data['Counts'], label='Counts')
        self._format_axes(self.axes, settings)

    def _format_axes(self, axes, graph):
        """Format the axes"""
        height_fractions = {'legend': 35, 'tick_label': 30, 'title': 20}
        height_fractions.update(graph.get('height_fractions', {}))
        self._format_axes_common(self.axes, graph, height_fractions)

        # Legends
        label_size = graph.get('height_fractions', {}).get('label_size', self._subplot_y_size // 26)
        axes.set_ylabel('Counts [cps]', fontsize=label_size)
        axes.set_xlabel('Binding energy [eV]', fontsize=label_size)


class Compare(BaseXPSFigure):
    """A figure type for comparisons of similar spectra"""

    def __init__(self, settings, data_type='avantage_export',
                 figure_args={'figsize': (18, 11), 'dpi': 100}):

        self._subplot_x_size = figure_args['figsize'][0] * figure_args['dpi']
        self._subplot_y_size = figure_args['figsize'][1] * figure_args['dpi']

        self.figure = plt.figure(**figure_args)
        self.axes = self.figure.add_subplot(111)

        if data_type == 'avantage_export':
            self._plot_from_avantage_export(settings)
        else:
            raise ValueError('Unkown data type: {}'.format(data_type))

        self._finalize_figure(settings)

    def _plot_from_avantage_export(self, settings):
        """Make the comparison plot from an avantage export"""
        autoscale_params = None
        for graph in settings['graphs']:
            data = graph['data']
            x = data[graph['key']]['Binding Energy (E)']
            y = data[graph['key']]['Counts']
            if graph.get('x_shift') is not None:
                x += graph['x_shift']

            legend = graph['legend']
            if 'auto_scale' in settings:
                base, height = self._calculate_base_and_height(x, y, settings['auto_scale'])

                if autoscale_params is None:
                    print('Save autoscale params based on', graph['legend'])
                    autoscale_params = base, height
                    legend = graph['legend']
                else:
                    y /= height / autoscale_params[1]
                    # Recalculate base after scaling
                    base, _ = self._calculate_base_and_height(x, y, settings['auto_scale'])
                    y -= base - autoscale_params[0]
                    legend = graph['legend'] + ' autoscale'

            self.axes.plot(x, y, label=legend)

        graph = settings.copy()
        graph['key'] = settings['graphs'][0]['key']

        self._format_axes(self.axes, graph)

    def _calculate_base_and_height(self, x, y, autoscale_settings):
        """Calculate the baseline"""
        auto_scale_limits = autoscale_settings['lim']
        revx = x[::-1]
        revy = y[::-1]
        cut_index = [np.searchsorted(revx, lim) for lim in auto_scale_limits]
        #print('cut_index', x[cut_index])

        npoints = autoscale_settings['npoints']
        #print('Lower', revx[cut_index[0]: cut_index[0] + npoints])
        #print('upper', revx[cut_index[1] - npoints: cut_index[1]])
        lower = revy[cut_index[0]: cut_index[0] + npoints].mean()
        upper = revy[cut_index[1] - npoints: cut_index[1]].mean()
        base_around = autoscale_settings['base_around']
        if base_around == 'upper':
            base = upper
        elif base_around == 'lower':
            base = lower
        elif base_around == 'both':
            base = np.mean([lower, upper])

        height = revy[cut_index[0]: cut_index[1]].max() - base

        return base, height

    def _format_axes(self, axes, graph):
        """Format the axes"""
        height_fractions = {'legend': 35, 'tick_label': 30, 'title': 20}
        height_fractions.update(graph.get('height_fractions', {}))
        self._format_axes_common(self.axes, graph, height_fractions)

        # Legends
        label_size = graph.get('height_fractions', {}).get('label_size', self._subplot_y_size // 26)
        axes.set_ylabel(graph.get('ylabel', 'Counts [cps]'), fontsize=label_size)
        axes.set_xlabel('Binding energy [eV]', fontsize=label_size)

    
        



def main():
    from PyExpLabSys.file_parsers.avantage_xlsx_export import AvantageXLSXExport
    avexport = AvantageXLSXExport("/home/kenni/Dokumenter/xps/soren_dahl_battery/soren_dahl_battery/HT1/peak table.xlsx")

    graphs = [
        {'key': 'Ni2p Scan', 'grid_pos': (0, 0), 'backgnd_label': 'Ni2p3 backgnd.', 'ylim': (2450, None)},
        {'key': 'Mn2p Scan', 'grid_pos': (0, 1), 'backgnd_label': 'Mn2p3 backgnd.', 'ylim': (None, 3300)},
        {'key': 'S2p Scan more scans', 'grid_pos': (1, 0), 'xlim': (160, None), 'ylim': (320, 380)},
        {'key': 'O1s Scan', 'grid_pos': (1, 1)},
        {'key': 'C1s Scan', 'grid_pos': (2, 0)},
        {'key': 'Si2p Scan', 'grid_pos': (2, 1)},
        
    ]
    settings = {
        'legend_fontsize': 'x-small',
        'gridspec': (3, 2),
        'graphs': graphs,
    }
    mulfig = MultipleFigures(avexport, settings)
    mulfig.show()


if __name__ == '__main__':
    main()
