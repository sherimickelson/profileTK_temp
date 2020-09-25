from pyinstrument import Profiler
from line_profiler import LineProfiler
import memory_profiler
import tracemalloc

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config

import pandas as pd
import numpy as np
from statistics import mean
import os
import linecache

class ProfileTK:
    """
    ProfileTK (a profiling toolkit) creates a performance object

    This package combines pyinstrument, line_profiler, memory_profiler, tracemalloc, and pycallgraph
    in order to provide a seemless and constant interface between them.  We provide timing profiles
    that can be analyized across different parameters, standalone timing profiles by function call
    and line by line, memory profiles by function call and line by line, and call graphs color coded 
    by runtime costs.

    """

    def __init__(self):
        """
        Initialize 

        Variables
        ---------
        profiles : dict
            The full profiles of each pyinstrument call within the collect_functional_times function.  
            Key: the key argument passed to collect_functional_times.  Value:  The text profile.
        timingsDF : pandas.DataFrame
            The profiling information gathered by calling pyinstrument within the collect_functional_times
            function.  Columns consist of the key and function names.
        Profiler : pyinstrument.Profiler
            The Profiler object within pyinstument.
        index : int, private
            A running index for the timingsDF dataframe.

        """

        self.profiles = {}
        self.timingsDF = pd.DataFrame()
        self._profiler = Profiler()
        self._index = 0


    def collect_functional_times(self, key: str, func: callable, *args, **kwargs):
        """
        Profiles a function with pyinstrument and populate the new timings into the dataframe.

        Parameters
        ----------
        key : str
            The unique identifier asigned to this new entry within the dataframe.
        func : callable
            The function to profile. 
        args : list, optional
            A list of arguments to pass to the function that is being profiled.
        kwargs : dict, optional
            A dictionary of keyword, value arguments to pass to the function that is being profiled.

        """

        # start profiler, run function, and stop profiler
        self._profiler.start()
        func(*args, **kwargs)
        self._profiler.stop()

        timers = {}
        local_timings = {}        

        # get profile results in a text format
        results = (self._profiler.output_text(unicode=True, show_all=True))
 
        # store the profile for future inqueries 
        self.profiles[key] = results

        # parse the profile into a dictionary
        start = False
        for line in results.splitlines():
            if start and 'module' not in line and 'self' not in line and '__' not in line and len(line.split())>3:
                ls = line.split('─ ')[1]
                t = ls.split()[1]
                if t in timers.keys():
                    timers[t] += float(ls.split()[0])
                else:
                    timers[t] = float(ls.split()[0])
            if ('Program:' in line):
                start = True
        
        local_timings[self._index] = {
            'key': key
        }
        for k,v in timers.items():
            local_timings[self._index][k] = v

        self._index += 1

        # store prfile into the dataframe
        df = pd.DataFrame(local_timings).T
        self.timingsDF = self.timingsDF.append(df).reset_index(drop=True).fillna(0.0)


    def print_timer_hotspots(self, key: str, l=25):
        """
        Prints the function names within the specified profile in descenting oder based on run time.
        This function prints the rank, the run time, the function name, file name and line number, and 
        the call tree depth.  This function uses the timing information provided by pyinstrument gathered 
        from the call to collect_functional_times. 

        Parameters
        ----------
        key : str
            The unique identifier of a profile that was set within the call to collect_functional_times.
        l : int
            The number of functions to print.
 
        """
        
        values = {} 
        # parse the saved profile to determine call tree depth and timings of each function
        for line in self.profiles[key].splitlines():
            if '─ ' in line:
                depth = int((len(line.split('─ ')[0])-1)/3)
                values[line.split('─ ')[1]+' call tree depth: '+str(depth)] = float(line.split('─ ')[1].split()[0])
        # sort into descending order and print
        sorted_values = sorted(values, reverse=True)[:l]
        for i,val in enumerate(sorted_values,1):
            print("#%s: %s" % (i,val))


    def collect_linebyline_times(self, func: callable, o_funcs: list, *args, **kwargs):
        """
        Calls the line_profiler utility to collect line by line timing information for specified functions.  
        New information is gathered each time this function is called so some variability exists between calls. 

        Parameters
        ----------
        func : callable
            The function to profile. 
        o_funcs : list
            A list of additional functions to profile.  These must be in the call tree of the function
            specified in the func argument.
        args : list, optional
            A list of arguments to pass to the function that is being profiled.
        kwargs : dict, optional
            A dictionary of keyword, value arguments to pass to the function that is being profiled.

        """

        lp = LineProfiler()
        for f in o_funcs:
            lp.add_function(f)
        lp_wrapper = lp(func)
        lp_wrapper(*args, **kwargs)
        lp.print_stats()


    def print_memory_hotspots(self, func: callable, l=10, *args, **kwargs):
        """
        Prints the memory hotspots within the selected function by calling tracemalloc in descending order.  
        This function prints the rank, the location of the code, size, and line of code.  New information
        is gathered each time this function is called so some variability may exist between calls.  

        Parameters
        ----------
        func : callable
            The function to profile. 
        l : int
            The number of hotspots to print.
        args : list, optional
            A list of arguments to pass to the function that is being profiled.
        kwargs : dict, optional
            A dictionary of keyword, value arguments to pass to the function that is being profiled.         

        """

        # start the profile, run the function, stop the profile
        tracemalloc.start()
        func(*args, **kwargs)
        trace = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # filter out calls we don't want to report
        trace = trace.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))

        # get the trace
        top = trace.statistics('lineno')
        
        # format the print for the selected number of lines to report
        for i,stat in enumerate(top[:l],1):
            frame = stat.traceback[0]
            print("#%s: %s:%s: %.1f KiB" % (i, frame.filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('     '+line)

        # print the total amount allocated to show relational value
        total = sum(stat.size for stat in top)
        print("Total amount allocated: %.1f KiB" % (total / 1024))


    def collect_linebyline_memory_usage(self, func: callable, o_funcs: list, *args, **kwargs):
        """
        Calls the memory_profiler utility to collect line by line memory information for specified functions.  
        New information is gathered each time this function is called so some variability exists between calls. 

        Parameters
        ----------
        func : callable
            The function to profile. 
        o_funcs : list
            A list of additional functions to profile.  These must be in the call tree of the function
            specified in the func argument.
        args : list, optional
            A list of arguments to pass to the function that is being profiled.
        kwargs : dict, optional
            A dictionary of keyword, value arguments to pass to the function that is being profiled.

        
        """

        prof = memory_profiler.LineProfiler(backend='psutil')
        for f in o_funcs:
            prof.add_function(f)
        val = prof(func)(*args, **kwargs)
        memory_profiler.show_results(prof, stream=None, precision=1)   


    def collect_memory_usage(self, func: callable, *args, **kwargs):
        """
        Calls the memory_profiler utility to collect the high water memory information for the specified function.  
        New information is gathered each time this function is called so some variability exists between calls. 

        Parameters
        ----------
        func : callable
            The function to profile. 
        args : list, optional
            A list of arguments to pass to the function that is being profiled.
        kwargs : dict, optional
            A dictionary of keyword, value arguments to pass to the function that is being profiled.

 
        """

        mem_usage = memory_profiler.memory_usage((func, args, kwargs))
        print (mean(mem_usage))

    
    def show_call_graph(self, func: callable, fn: str, *args, **kwargs):
        """
        Creates a call graph for the speficed function that is color coded to indicate which functions
        take longer to run.

        This requires an install of graphviz (https://www.graphviz.org/).  

        Parameters
        ----------
        func : callable
            The function to profile. 
        fn : str
            The name of the filename to save the graph image to. 
        args : list, optional
            A list of arguments to pass to the function that is being profiled.
        kwargs : dict, optional
            A dictionary of keyword, value arguments to pass to the function that is being profiled.

        """

        graphviz = GraphvizOutput()
        graphviz.output_file = fn
        config = Config()
        config.include_stdlib = True

        with PyCallGraph(config=config, output=graphviz):
            func(*args, **kwargs)


 
