from ahm import *
from dfm import *

set_log_level(LogLevel.WARNING)

def run(N, n, h, k, c, g, f, ls='mumps', pc='default', do_write=True):
    t1 = Timer()
    m = int(N/n)
    l = int(m/2)
    fp, fm, fd, fb = fibers(N, N, l, h, n, k*n, do_write)
    tp, tm, td, tb = fibers(N, N, 0, 0, n, k*n, do_write)
    lp, lm, ld, lb = fibers(int(m/c), int(m/c), int(l/c), int(h/c), 1, k, do_write)
    cp, cm, cd, cb = fibers(int(m/c), int(m/c), int(l/c/n), int(h/c/n), n, k*n, do_write)
    E = [20e9, f*20e9]
    mu, lmd = lame(E, [0.3, 0.3])
    t1.stop()
    
    t2 = Timer()
    u = fem(fp, fm, fd, fb, mu, lmd, g, ls, pc, do_write)
    ul2 = norm(u)
    uli = norm(u.vector(), 'linf')
    t2.stop()

    t3 = Timer()
    ua = ahm(lp + 'ahm_', lm, ld, lb, cp + 'ahm_', cm, cd, cb, mu, lmd, g, ls, pc, do_write)
    ea = project(ua - u, u.function_space(), solver_type=ls, preconditioner_type=pc)
    eal2 = norm(ea)/ul2
    eali = norm(ea.vector(), 'linf')/uli
    if do_write:
        XDMFFile(fp + 'ahm_error.xdmf').write(ea)
    t3.stop()

    t4 = Timer()
    ud = dfm(cp + 'dfm_', cm, cd, cb, mu, lmd, g, h/N, ls, pc, do_write)
    ed = project(ud - u, u.function_space(), solver_type=ls, preconditioner_type=pc)
    edl2 = norm(ed)/ul2
    edli = norm(ed.vector(), 'linf')/uli
    if do_write:
        XDMFFile(fp + 'dfm_error.xdmf').write(ed)
    t4.stop()

    # print(t1.elapsed()[0], t2.elapsed()[0], t3.elapsed()[0], t4.elapsed()[0])
    return eal2, eali, edl2, edli


# for ls in ['bicgstab', 'cg', 'gmres', 'minres', 'tfqmr']:
#     for pc in ['amg', 'hypre_amg', 'hypre_euclid', 'hypre_parasails']:
#         timer = Timer()
#         try:
#             eal2, eali, edl2, edli = run(256, 1, 2, 2, 1, ls, pc, False)
#             print(ls, pc, timer.elapsed(), eal2, eali, edl2, edli)
#         except:
#             print(ls, pc, timer.elapsed(), 'error')

ls, pc, do_write = 'cg', 'amg', False

N0, n0, k0, d0, c0, f0 = 2048, 4, 1, 2, 1, 32


for g in [Constant([0, -1e5])]:
    # run(N0, n0, d0, k0, c0, g, f0, ls, pc, True)
    print('k\teal2\teali\tedl2\tedli')
    for k in [1, 2, 4, 8, 16]:
        eal2, eali, edl2, edli = run(N0, n0, d0, k, c0, g, f0, ls, pc, do_write)
        print(f'{k*n0*n0}\t{eal2}\t{eali}\t{edl2}\t{edli}')
    print('d\teal2\teali\tedl2\tedli')
    for d in [2, 4, 8, 16, 32]:
        eal2, eali, edl2, edli = run(N0, n0, d, k0, c0, g, f0, ls, pc, do_write)
        print(f'{d/N0}\t{eal2}\t{eali}\t{edl2}\t{edli}')
    print('f\teal2\teali\tedl2\tedli')
    for f in [8, 16, 32, 64, 128]:
        eal2, eali, edl2, edli = run(N0, n0, d0, k0, c0, g, f, ls, pc, do_write)
        print(f'{f}\t{eal2}\t{eali}\t{edl2}\t{edli}')
    print('n\teal2\teali\tedl2\tedli')
    for n in [1, 2, 4, 8, 16]:
        eal2, eali, edl2, edli = run(int(N0*n/16), n, d0, k0, c0, g, f0, ls, pc, do_write)
        print(f'{n}\t{eal2}\t{eali}\t{edl2}\t{edli}')
    print('h\teal2\teali\tedl2\tedli')
    for c in [16, 8, 4, 2, 1]:
        eal2, eali, edl2, edli = run(N0, n0, 16*d0, k0, c, g, f0, ls, pc, do_write)
        print(f'{int(c/N0)}\t{eal2}\t{eali}\t{edl2}\t{edli}')