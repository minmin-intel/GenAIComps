# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

def setup_memory(args):
    if args.memory_type == "sqlite":
        return SqliteSaver.from_conn_string(":memory:")
    elif args.memory_type == "in-memory":
        return MemorySaver()