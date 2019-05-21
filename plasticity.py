from fem import *

def mises(u, mu, lmd):
    s = sigma(u, mu, lmd)
    return 1.0/6.0*((s[0,0]-s[1,1])**2 + s[0,1]**2)

def plasticity(path, mesh, domains, bounds, mu, lmd, g, ls='mumps', pc='default', do_write=True):
    dx = Measure('dx', mesh, subdomain_data=domains)
    ds = Measure('ds', mesh, subdomain_data=bounds)
    V = VectorFunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = Function(V)

    mt = 1e10
    m  = mises(u, mu[0], lmd[0])/mt
    dm = 1.0
    w = 1.0/(1.0 + exp(1.0/dm*(1.0-m)))
    # w = conditional(le(m, 1-dm), 0, conditional(ge(m, 1+dm), 1, (m-1-dm)/2/dm))
    muw = mu[0]*(1.0-w) + mu[1]*w
    lmdw = lmd[0]*(1.0-w) + lmd[1]*w

    F = inner(sigma(u, muw, lmdw), sym(grad(v)))*dx - inner(g, v)*ds(2)
    bcs = [DirichletBC(V, Constant((0, 0)), bounds, 1)]
    solve(F == 0, u, bcs, solver_parameters={'newton_solver': {'linear_solver': ls, 'preconditioner': pc, 'relaxation_parameter': 1.0}})
    if do_write:
        XDMFFile(f'{path}u.xdmf').write(u)
        W = FunctionSpace(mesh, 'DG', 0)
        XDMFFile(f'{path}mu.xdmf').write(project(muw, W))
        XDMFFile(f'{path}lmd.xdmf').write(project(lmdw, W))
    return u

if __name__ == '__main__':
    p, m, d, b = fibers(128, 128, 16, 2, 4, 8)
    mu, lmd = lame([20e9, 20e9], [0.3, 0.3])
    u = fem(p, m, d, b, mu, lmd, Constant([0, -1e5]), 'cg', 'amg')
    s = stress(p, m, d, b, mu, lmd, u)
    XDMFFile(f'{p}mises.xdmf').write(project(mises(u, mu[0], lmd[0]), FunctionSpace(m, 'DG', 0)))
    mup, lmdp = lame([20e9, 10e9], [0.3, 0.3])
    u = plasticity(p + 'p', m, d, b, mup, lmdp, Constant([0, -1e5]), 'gmres', 'amg')
    s = stress(p, m, d, b, mu, lmd, u)
    XDMFFile(f'{p}pmises.xdmf').write(project(mises(u, mu[0], lmd[0]), FunctionSpace(m, 'DG', 0)))
