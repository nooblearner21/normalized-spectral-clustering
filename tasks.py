from invoke import task


@task
def run(ctx, k=0, n=0, Random=True):
    ctx.run('python setup.py build_ext --inplace')
    if(Random):
        ctx.run('python main.py {} {} --Random'.format(k, n))
    else:
        ctx.run('python main.py {} {} --no-Random'.format(k, n))

