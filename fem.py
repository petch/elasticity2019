from domain import *

def lame(E, nu):
    mu  = [Constant(E[i]/(2*(1 + nu[i])))                   for i in range(len(E))]
    lmd = [Constant(E[i]*nu[i]/((1 + nu[i])*(1 - 2*nu[i]))) for i in range(len(E))]
    return mu, lmd

def sigma(v, mu, lmd):
    return 2.0*mu*sym(grad(v)) + lmd*tr(sym(grad(v)))*Identity(len(v))

def stress(path, mesh, domains, bounds, mu, lmd, u, ls='mumps', pc='default', do_write=True):
    dx = Measure('dx', mesh, subdomain_data=domains)
    ds = Measure('ds', mesh, subdomain_data=bounds)
    W = TensorFunctionSpace(mesh, "DG", 0)
    s = TrialFunction(W)
    w = TestFunction(W)
    a = inner(s, w)*dx
    L = sum([inner(sigma(u, mu[i], lmd[i]), w)*dx(i+1) for i in range(len(mu))])
    s = Function(W)
    solve(a == L, s, solver_parameters={'linear_solver': ls, 'preconditioner': pc})
    if do_write:
        XDMFFile(f'{path}s.xdmf').write(s)
    return s

def fem(path, mesh, domains, bounds, mu, lmd, g, ls='mumps', pc='default', do_write=True):
    dx = Measure('dx', mesh, subdomain_data=domains)
    ds = Measure('ds', mesh, subdomain_data=bounds)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = sum([inner(sigma(u, mu[i], lmd[i]), sym(grad(v)))*dx(i+1) for i in range(len(mu))])
    L = inner(g, v)*ds(2)
    bcs = [DirichletBC(V, Constant((0, 0)), bounds, 1)]
    u = Function(V)
    solve(a == L, u, bcs, solver_parameters={'linear_solver': ls, 'preconditioner': pc})
    if do_write:
        XDMFFile(f'{path}u.xdmf').write(u)
    return u

if __name__ == '__main__':
    p, m, d, b = fibers(128, 128, 16, 2, 4, 8)
    mu, lmd = lame([20e9, 20e9], [0.3, 0.3])
    u = fem(p, m, d, b, mu, lmd, Constant([0, -1e5]), 'cg', 'amg')
    s = stress(p, m, d, b, mu, lmd, u)
    
