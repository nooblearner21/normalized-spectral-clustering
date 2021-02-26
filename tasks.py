from invoke import task


@task
def run(ctx, k=0, n=0, random=True):
    ctx.run('python setup.py build_ext --inplace')
    if(random):
        ctx.run('python main.py {} {} --random'.format(k, n))
    else:
        ctx.run('python main.py {} {} --no-random'.format(k, n))

