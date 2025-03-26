from executorlib import SingleNodeExecutor


def calc(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank



with SingleNodeExecutor(resource_dict={"cores": 3}) as exe:
    fs = exe.submit(calc, 3)
    print(fs.result())

