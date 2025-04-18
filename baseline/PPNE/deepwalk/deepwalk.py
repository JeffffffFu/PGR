#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

# from . import graph
from . import graph
# from . import walks as serialized_walks
from . import walks as serialized_walks
from gensim.models import Word2Vec
from .skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

from . import weighted_random_walk
import psutil
from multiprocessing import cpu_count
import baseline.PPNE.util

from pdb import set_trace as st

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def deepwalk(args, g, verbose=True, random_seed=0):

  G = g
  if verbose:
    print("Random seed used %d"%random_seed)
  np.random.seed(random_seed)

  if verbose:
    print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.num_walks

  if verbose:
    print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  if verbose:
    print("Data size (walks*length): {}".format(data_size))

  if data_size < args.max_memory_data_size:
    if verbose:
      print("Walking...")
    # walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                        # path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
    walks = weighted_random_walk.random_walk(G, num_paths=args.num_walks,path_length=args.walk_length, alpha=0)

    if verbose:
      print("Training...")
    model = Word2Vec(walks, vector_size=args.dim, window=args.window_size, seed=random_seed, min_count=0, sg=1, hs=0,
                     epochs=args.iter, workers=args.workers)
  else:
    if verbose:
      print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
      print("Walking...")

    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.num_walks,
                                         path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                         num_workers=args.workers)

    if verbose:
      print("Counting vertex frequency...")
    if not args.vertex_freq_degree:
      vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
    else:
      # use degree distribution for frequency in tree
      vertex_counts = G.degree(nodes=G.iterkeys())

    if verbose:
      print("Training...")
    walks_corpus = serialized_walks.WalksCorpus(walk_files)
    model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                     size=args.dim,
                     window=args.window_size, min_count=0, iter=args.iter, trim_rule=None, workers=args.workers)

  # model.wv.save_word2vec_format(args.output)

  emb_X = []
  emb_Y = []
  vocab = model.wv.key_to_index
  for node_index in range(0, len(G.nodes())):
      node_str = str(node_index)
      index = vocab.get(node_str, None)
      node_emb = model.wv[node_str]
      context_emb = model.syn1neg[index]
      emb_X.append(node_emb)
      emb_Y.append(context_emb)
  X = np.vstack(emb_X)
  Y = np.vstack(emb_Y)

  return X, Y

def deepwalk_v2(args, g, verbose=True, random_seed=0):
  G = graph.from_networkx(g)
  # G = g
  if verbose:
    print("Random seed used %d"%random_seed)
  np.random.seed(random_seed)

  if verbose:
    print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.num_walks

  if verbose:
    print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  if verbose:
    print("Data size (walks*length): {}".format(data_size))

  if data_size < args.max_memory_data_size:
    if verbose:
      print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=args.num_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
    # walks = weighted_random_walk.random_walk(G, num_paths=args.num_walks,path_length=args.walk_length, alpha=0)

    if verbose:
      print("Training...")
    model = Word2Vec(walks, vector_size=args.dim, window=args.window_size, seed=random_seed, min_count=0, sg=1, hs=0,
                     epochs=args.iter, workers=args.workers)
  else:
    if verbose:
      print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
      print("Walking...")

    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.num_walks,
                                         path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                         num_workers=args.workers)

    if verbose:
      print("Counting vertex frequency...")
    if not args.vertex_freq_degree:
      vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
    else:
      # use degree distribution for frequency in tree
      vertex_counts = G.degree(nodes=G.iterkeys())

    if verbose:
      print("Training...")
    walks_corpus = serialized_walks.WalksCorpus(walk_files)
    model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                     size=args.dim,
                     window=args.window_size, min_count=0, iter=args.iter, trim_rule=None, workers=args.workers)

  # model.wv.save_word2vec_format(args.output)

  emb_X = []
  emb_Y = []
  vocab = model.wv.key_to_index
  for node_index in range(0, len(G.nodes())):
      node_str = str(node_index)
      index = vocab.get(node_str, None)
      node_emb = model.wv[node_str]
      context_emb = model.syn1neg[index]
      emb_X.append(node_emb)
      emb_Y.append(context_emb)
  X = np.vstack(emb_X)
  Y = np.vstack(emb_Y)

  return X, Y

def main():
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output', required=True,
                      help='Output representation file')

  parser.add_argument('--representation-size', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')


  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  process(args)

if __name__ == "__main__":
  sys.exit(main())
