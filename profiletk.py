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
    """

    def __init__(self):
        """
        
        """

        self.profiles = {}
        self.timingsDF = pd.DataFrame()
        self._profiler = Profiler()
        self._index = 0


    def collect_functional_times(self, key: str, func: callable, **kwargs):
        """
        
        """
        self._profiler.start()
        func(**kwargs)
        self._profiler.stop()

        timers = {}
        local_timings = {}        

        results = (self._profiler.output_text(unicode=True, show_all=True))
 
        self.profiles[key] = results

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

        df = pd.DataFrame(local_timings).T
        self.timingsDF = self.timingsDF.append(df).reset_index(drop=True).fillna(0.0)


    def print_timer_hotspots(self, key: str, l=25):
        """
        
        """
        
        values = {} 
        for line in self.profiles[key].splitlines():
            if '─ ' in line:
                depth = int((len(line.split('─ ')[0])-1)/3)
                values[line.split('─ ')[1]+' call tree depth: '+str(depth)] = float(line.split('─ ')[1].split()[0])
        sorted_values = sorted(values, reverse=True)[:l]
        for i,val in enumerate(sorted_values,1):
            print("#%s: %s" % (i,val))


    def collect_linebyline_times(self, func: callable, o_funcs: list, **kwargs):
        """
        
        """
        lp = LineProfiler()
        for f in o_funcs:
            lp.add_function(f)
        lp_wrapper = lp(func)
        lp_wrapper(**kwargs)
        lp.print_stats()


    def print_memory_hotspots(self, func: callable, l=10, **kwargs):
        """
        
        """
        tracemalloc.start()
        func(**kwargs)
        trace = tracemalloc.take_snapshot()
        tracemalloc.stop()

        trace = trace.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top = trace.statistics('lineno')
        
        for i,stat in enumerate(top[:l],1):
            frame = stat.traceback[0]
            print("#%s: %s:%s: %.1f KiB" % (i, frame.filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('     '+line)
        
        total = sum(stat.size for stat in top)
        print("Total amount allocated: %.1f KiB" % (total / 1024))


    def collect_linebyline_memory_usage(self, func: callable, o_funcs: list, **kwargs):
        """
        
        """
        prof = memory_profiler.LineProfiler(backend='psutil')
        for f in o_funcs:
            prof.add_function(f)
        val = prof(func)(**kwargs)
        memory_profiler.show_results(prof, stream=None, precision=1)   


    def collect_memory_usage(self, func: callable, **kwargs):
        """
        
        """
        mem_usage = memory_profiler.memory_usage((func, [], kwargs))
        print (mean(mem_usage))

    
    def show_call_graph(self, func: callable, fn: str, **kwargs):
        """
        This requires an install of graphviz (https://www.graphviz.org/)  
        """
        graphviz = GraphvizOutput()
        graphviz.output_file = fn
        config = Config()
        config.include_stdlib = True

        with PyCallGraph(config=config, output=graphviz):
            func(**kwargs)


 
