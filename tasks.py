from invoke import task


@task
def run(ctx, k=0, n=0, random=True):
    ctx.run('python3.8.5 setup.py build_ext --inplace')
    if(random):
        ctx.run('python3.8.5 main.py {} {} --random'.format(k, n))
    else:
        ctx.run('python3.8.5 main.py {} {} --no-random'.format(k, n))

