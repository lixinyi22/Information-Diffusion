# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils
from utils.load import DynamicCasHypergraph
from helper.BaseReader import BaseReader


class MINDSReader(BaseReader):
    def __init__(self, args):
        super().__init__(args)
        
        graph_path = os.path.join(args.path, args.dataset)
        self.diffusion_graph = DynamicCasHypergraph(graph_path, device=args.device)