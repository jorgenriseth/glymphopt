import scripts.dfa as dfa


class Domain(dfa.Mesh):
    def __init__(
        self,
        mesh: dfa.Mesh,
        subdomains: dfa.MeshFunction,
        boundaries: dfa.MeshFunction,
        **kwargs,
    ):
        super().__init__(mesh, **kwargs)
        self.subdomains = transfer_meshfunction(self, subdomains)
        self.boundaries = transfer_meshfunction(self, boundaries)


def transfer_meshfunction(
    newmesh: dfa.Mesh, meshfunc: dfa.MeshFunction
) -> dfa.MeshFunction:
    newtags = dfa.MeshFunction("size_t", newmesh, dim=meshfunc.dim())  # type: ignore
    newtags.set_values(meshfunc)  # type: ignore
    return newtags
