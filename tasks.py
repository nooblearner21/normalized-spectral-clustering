from invoke import task


@task
def run(ctx, k=0, n=0, Random=True):
    ctx.run('python3.8.5 setup.py build_ext --inplace')
    if(Random):
        ctx.run('python3.8.5 main.py {} {} --Random'.format(k, n))
    else:
        ctx.run('python3.8.5 main.py {} {} --no-Random'.format(k, n))

