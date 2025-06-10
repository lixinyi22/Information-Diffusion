# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils
from utils.load import ConHypergraph
from helper.BaseReader import BaseReader


class DisenIDPReader(BaseReader):
    def __init__(self, args):
        super().__init__(args)
        
        graph_path = os.path.join(args.path, args.dataset)
        self.diffusion_graph = ConHypergraph(graph_path)