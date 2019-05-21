from dolfin import *

def fibers(N, M, l, h, n=1, m=1, do_write=True):
    mesh = UnitSquareMesh(N, M)
    sx = (N/n - l)/N/2
    ex = sx + l/N
    sy = (M/m - h)/M/2
    ey = sy + h/M

    domains = MeshFunction('size_t', mesh, 2, 1)
    CompiledSubDomain(f'\
        between(x[0], std::make_pair(floor(x[0]*{n})/{n} + {sx}, floor(x[0]*{n})/{n} + {ex})) &&\
        between(x[1], std::make_pair(floor(x[1]*{m})/{m} + {sy}, floor(x[1]*{m})/{m} + {ey}))'
    ).mark(domains, 2)

    bounds = MeshFunction('size_t', mesh, 1)
    CompiledSubDomain('near(x[0], 0)').mark(bounds, 1)
    CompiledSubDomain('near(x[0], 1)').mark(bounds, 2)
    CompiledSubDomain('near(x[1], 0) && x[0] < 2.0/8.0 + DOLFIN_EPS && x[0] > 1.0/8.0 - DOLFIN_EPS').mark(bounds, 3)
    CompiledSubDomain('near(x[1], 1) && x[0] > 7.0/8.0 - DOLFIN_EPS').mark(bounds, 4)
    CompiledSubDomain(f'\
        between(x[0], std::make_pair(floor(x[0]*{n})/{n} + {sx}, floor(x[0]*{n})/{n} + {ex})) &&\
        near(x[1], floor(x[1]*{m})/{m} + {(sy + ey)/2})'
    ).mark(bounds, 5)

    path = f"results/{N}x{M}h{h}l{l}n{n}m{m}/"

    if do_write:
        File(f'{path}mesh.xml').write(mesh)
        File(f'{path}domains.xml').write(domains)
        File(f'{path}bounds.xml').write(bounds)

        XDMFFile(f'{path}mesh.xdmf').write(mesh)
        XDMFFile(f'{path}domains.xdmf').write(domains)
        XDMFFile(f'{path}bounds.xdmf').write(bounds)

    return path, mesh, domains, bounds

if __name__ == '__main__':
    p, m, d, b = fibers(128, 128, 16, 2, 4, 8)