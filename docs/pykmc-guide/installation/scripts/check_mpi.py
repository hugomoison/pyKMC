#!/usr/bin/env python3
"""Two-rank MPI communication check; no atomic simulation or output files."""

from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size != 2:
        raise RuntimeError(f"Launch exactly two MPI ranks; got {size}")
    rank = comm.Get_rank()
    ranks = comm.allgather(rank)
    total = comm.allreduce(rank + 1, op=MPI.SUM)
    token = comm.bcast("pykmc-mpi-check" if rank == 0 else None, root=0)
    if ranks != [0, 1] or total != 3 or token != "pykmc-mpi-check":
        raise RuntimeError("MPI collective communication check failed")
    if rank == 0:
        print("PASS: 2 MPI ranks; allgather, allreduce and broadcast")


if __name__ == "__main__":
    main()
