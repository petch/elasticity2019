from local import *

def locals(path, mesh, domains, bounds, mu, lmd, local_solver, ls='mumps', pc='default', do_write=True):
    dx = Measure('dx', mesh, subdomain_data=domains)
    ds = Measure('ds', mesh, subdomain_data=domains)
    x1 = Expression(('x[0]', '0'), domain=mesh, degree=1)
    x2 = Expression(('x[1]/2', 'x[0]/2'), domain=mesh, degree=1)
    x3 = Expression(('0', 'x[1]'), domain=mesh, degree=1)
    u1 = local_solver(path + '1_', mesh, domains, bounds, mu, lmd, x1, ls, pc, do_write)
    s1 = stress(path + '1_', mesh, domains, bounds, mu, lmd, u1 + x1, ls, pc, do_write)
    u2 = local_solver(path + '2_', mesh, domains, bounds, mu, lmd, x2, ls, pc, do_write)
    s2 = stress(path + '2_', mesh, domains, bounds, mu, lmd, u2 + x2, ls, pc, do_write)
    u3 = local_solver(path + '3_', mesh, domains, bounds, mu, lmd, x3, ls, pc, do_write)
    s3 = stress(path + '3_', mesh, domains, bounds, mu, lmd, u3 + x3, ls, pc, do_write)
    E11 = assemble(s1[0, 0]*dx)
    E12 = assemble(s1[0, 1]*dx)
    E13 = assemble(s1[1, 1]*dx)
    E21 = assemble(s2[0, 0]*dx)
    E22 = assemble(s2[0, 1]*dx)
    E23 = assemble(s2[1, 1]*dx)
    E31 = assemble(s3[0, 0]*dx)
    E32 = assemble(s3[0, 1]*dx)
    E33 = assemble(s3[1, 1]*dx)
    E = ((E11, E21, E31), (E12, E22, E32), (E13, E23, E33))
    # print(f'E = {E}')
    return E, u1, u2, u3

def epsilon(v):
    return as_vector((v[0].dx(0), v[0].dx(1) + v[1].dx(0), v[1].dx(1)))

def coarse(path, mesh, domains, bounds, E, g, ls='mumps', pc='default', do_write=True):
    dx = Measure('dx', mesh, subdomain_data=domains)
    ds = Measure('ds', mesh, subdomain_data=bounds)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(dot(E, epsilon(u)), epsilon(v))*dx
    L = inner(g, v)*ds(2)
    bcs = [DirichletBC(V, Constant((0, 0)), bounds, 1)]
    u = Function(V)
    solve(a == L, u, bcs, solver_parameters={'linear_solver': ls, 'preconditioner': pc})
    if do_write:
        XDMFFile(f'{path}u.xdmf').write(u)
    return u

def ahmfine(local_solver, n, path, mesh, domains, bounds, lpath, lmesh, ldomains, lbounds, cpath, cmesh, cdomains, cbounds, mu, lmd, g, ls='mumps', pc='default', do_write=True):
    E, u1, u2, u3 = locals(lpath + 'local_', lmesh, ldomains, lbounds,  mu, lmd, local_solver, ls, pc, do_write)
    uc = coarse(cpath + 'coarse_', cmesh, cdomains, cbounds, Constant(E), g, ls, pc, do_write)
    u1 = Repeat(path + '1_', mesh, u1, n, ls, pc, do_write)
    u2 = Repeat(path + '2_', mesh, u2, n, ls, pc, do_write)
    u3 = Repeat(path + '3_', mesh, u3, n, ls, pc, do_write)
    W = FunctionSpace(cmesh, "DG", 0)
    du1 = project(uc[0].dx(0), W, solver_type=ls, preconditioner_type=pc)
    du2 = project(uc[0].dx(1) + uc[1].dx(0), W, solver_type=ls, preconditioner_type=pc)
    du3 = project(uc[1].dx(1), W, solver_type=ls, preconditioner_type=pc)
    ux = uc[0] - 1/n*(u1[0]*du1 + u2[0]*du2 + u3[0]*du3)
    uy = uc[1] - 1/n*(u1[1]*du1 + u2[1]*du2 + u3[1]*du3)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = project(as_vector((ux, uy)), V, solver_type=ls, preconditioner_type=pc)
    if do_write:
        XDMFFile(f'{path}u.xdmf').write(u)
    return u
    
def ahm(lpath, lmesh, ldomains, lbounds, cpath, cmesh, cdomains, cbounds, mu, lmd, g, ls='mumps', pc='default', do_write=True):
    E, u1, u2, u3 = locals(lpath + 'local_', lmesh, ldomains, lbounds,  mu, lmd, periodic, ls, pc, do_write)
    return coarse(cpath + 'coarse_', cmesh, cdomains, cbounds, Constant(E), g, ls, pc, do_write)

if __name__ == "__main__":
    # m, l, h, n, k = 32, 16, 2, 4, 2
    # fp, fm, fd, fb = fibers(n*m, n*m, l, h, n, k*n)
    # lp, lm, ld, lb = fibers(m, m, l, h, 1, k)
    # cp, cm, cd, cb = fibers(m, m, l/n, h/n, n, k*n)
    # mu, lmd = lame([20e9, 200e9], [0.3, 0.3])
    # g = Constant([0, -1e5])
    # u = fem(fp, fm, fd, fb, mu, lmd, g)
    # s = stress(fp, fm, fd, fb, mu, lmd, u)
    
    # ua = ahm(lp + 'ahm_', lm, ld, lb, cp + 'ahm_', cm, cd, cb, mu, lmd, g)
    # sa = stress(cp + 'ahm_', cm, cd, cb, mu, lmd, ua)
    # ea = project(u - ua, u.function_space())
    # print(norm(ea)/norm(u))
    # XDMFFile(cp + 'ahm_error.xdmf').write(ea)

    set_log_level(LogLevel.WARNING)
    for i in range(1, 21):
        m, l, h, n, k = 512, 256, 32, 1, 1
        lp, lm, ld, lb = fibers(m, m, l, h, 1, k)
        mu, lmd = lame([20e9, 10e9*i], [0.3, 0.3])
        E, u1, u2, u3 = locals(lp, lm, ld, lb, mu, lmd, periodic, 'cg', 'amg')
        print(10e9*i)
        print(E)