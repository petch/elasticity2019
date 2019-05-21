from fem import *

def sigman(v, mu, lmd):
    return (2.0*mu+lmd)*grad(v)

def dfm(path, mesh, domains, bounds, mu, lmd, g, h, ls='mumps', pc='default', do_write=True):
    dx = Measure('dx', mesh, subdomain_data=domains)
    ds = Measure('ds', mesh, subdomain_data=bounds)
    dS = Measure('dS', mesh, subdomain_data=bounds)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    n = FacetNormal(mesh)
    t = as_vector([n('+')[1], -n('+')[0]])
    un = inner(u('+'), t)
    vn = inner(v('+'), t)
    a = inner(sigma(u, mu[0], lmd[0]), sym(grad(v)))*dx\
      + Constant(h)*inner(inner(sigman(un, mu[1]-mu[0], lmd[1]-lmd[0]), t), inner(grad(vn), t))*dS(5)
    L = inner(g, v)*ds(2)
    bcs = [DirichletBC(V, Constant((0, 0)), bounds, 1)]
    u = Function(V)
    solve(a == L, u, bcs, solver_parameters={'linear_solver': ls, 'preconditioner': pc})
    if do_write:
        XDMFFile(f'{path}u.xdmf').write(u)
    return u

if __name__ == '__main__':
    p, m, d, b = fibers(128, 128, 16, 2, 4, 8)
    mu, lmd = lame([20e9, 200e9], [0.3, 0.3])
    g = Constant([0, -1e5])
    u = fem(p, m, d, b, mu, lmd, g)
    s = stress(p, m, d, b, mu, lmd, u)

    ud = dfm(p + 'dfm_', m, d, b, mu, lmd, Constant([0, -1e5]), Constant(2/128))
    sd = stress(p + 'dfm_', m, d, b, mu, lmd, u)
    ed = project(u - ud, u.function_space())
    print(norm(ed)/norm(u))
    XDMFFile(p + 'dfm_error.xdmf').write(ed)
