from fem import *

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not near(x[0], 1) and not near(x[1], 1)

    def map(self, x, y):
        if near(x[0], 1):
            y[0] = x[0] - 1
        else:
            y[0] = x[0]
        if near(x[1], 1):
            y[1] = x[1] - 1
        else:
            y[1] = x[1]

def periodic(path, mesh, domains, bounds, mu, lmd, x, ls='mumps', pc='default', do_write=True):
    dx = Measure('dx', mesh, subdomain_data=domains)
    ds = Measure('ds', mesh, subdomain_data=bounds)
    V = VectorFunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())
    u = TrialFunction(V)
    v = TestFunction(V)
    a = sum([inner(sigma(u, mu[i], lmd[i]), sym(grad(v)))*dx(i+1) for i in range(len(mu))])
    L = sum([-inner(sigma(x, mu[i], lmd[i]), sym(grad(v)))*dx(i+1) for i in range(len(mu))])
    bcs = [DirichletBC(V, Constant((0, 0)), 'near(x[0], 0.5) and near(x[1], 0.5)', method='pointwise')]
    u = Function(V)
    solve(a == L, u, bcs, solver_parameters={'linear_solver': ls, 'preconditioner': pc})
    if do_write:
        XDMFFile(f'{path}u.xdmf').write(u)
    return u

def dirichlet(path, mesh, domains, bounds, mu, lmd, x, ls='mumps', pc='default', do_write=True):
    dx = Measure('dx', mesh, subdomain_data=domains)
    ds = Measure('ds', mesh, subdomain_data=bounds)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = sum([inner(sigma(u, mu[i], lmd[i]), sym(grad(v)))*dx(i+1) for i in range(len(mu))])
    L = sum([-inner(sigma(x, mu[i], lmd[i]), sym(grad(v)))*dx(i+1) for i in range(len(mu))])
    bcs = [DirichletBC(V, Constant((0, 0)), 'on_boundary')]
    u = Function(V)
    solve(a == L, u, bcs, solver_parameters={'linear_solver': ls, 'preconditioner': pc})
    if do_write:
        XDMFFile(f'{path}u.xdmf').write(u)
    return u

repeat_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
#include <dolfin/common/Array.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
class Repeat : public dolfin::Expression {{
public:
    Repeat() : dolfin::Expression(2) {{ }}
    void eval(dolfin::Array< double > &values, const dolfin::Array< double > &x) const override {{
        dolfin::Array< double > y(2);
        y[0] = {0}*x[0] - floor({0}*x[0]);
        y[1] = {0}*x[1] - floor({0}*x[1]);
        (*u).eval(values, y);
    }}
    std::shared_ptr<dolfin::Function> u;
}};
PYBIND11_MODULE(SIGNATURE, m) {{
    py::class_<Repeat, std::shared_ptr<Repeat>, dolfin::Expression>(m, "Repeat")
        .def(py::init<>())
        .def_readwrite("u", &Repeat::u);
}}
"""
def Repeat(path, mesh, u, n, ls='mumps', pc='default', do_write=True):
    c = CompiledExpression(
        compile_cpp_code(repeat_code.format(n)).Repeat(),
        u=u, degree=1)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = project(c/n/n, V, solver_type=ls, preconditioner_type=pc)
    if do_write:
        XDMFFile(f'{path}repeat_u.xdmf').write(u)
    return u

if __name__ == '__main__':
    p, m, d, b = fibers(32, 32, 20, 2, 1, 2)
    mu, lmd = lame([20e9, 200e9], [0.3, 0.3])
    x = Expression(('x[0]', '0'), domain=m, degree=1)
    u = periodic(p + 'p_', m, d, b, mu, lmd, x)
    u = dirichlet(p + 'd_', m, d, b, mu, lmd, x)
    
    p, m, d, b = fibers(128, 128, 20, 2, 4, 8)
    u = Repeat(p, m, u, 4)
    
