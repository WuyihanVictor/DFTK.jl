# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/temperature)
# DOS (density of states)
# D(ε) = N'(ε)
# LDOS (local density of states)
# LD = sum_n f_n |ψn|^2

"""
Total density of states at energy ε
"""
function compute_dos(ε, basis, eigenvalues; smearing=basis.model.smearing,
                     temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_dos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    D = zeros(typeof(ε), basis.model.n_spin_components)
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            D[σ] -= (filled_occ * basis.kweights[ik] / temperature
                     * Smearing.occupation_derivative(smearing, enred))
        end
    end
    D = mpi_sum(D, basis.comm_kpts)
end
function compute_dos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
        compute_dos(ε, scfres.basis, scfres.eigenvalues; kwargs...)
end

"""
Local density of states, in real space. `weight_threshold` is a threshold
to screen away small contributions to the LDOS.
"""
function compute_ldos(ε, basis::PlaneWaveBasis{T}, eigenvalues, ψ;
                      smearing=basis.model.smearing,
                      temperature=basis.model.temperature,
                      weight_threshold=eps(T)) where {T}
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_ldos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    weights = deepcopy(eigenvalues)
    for (ik, εk) in enumerate(eigenvalues)
        for (iband, εnk) in enumerate(εk)
            enred = (εnk - ε) / temperature
            weights[ik][iband] = (-filled_occ / temperature
                                  * Smearing.occupation_derivative(smearing, enred))
        end
    end

    # Use compute_density routine to compute LDOS, using just the modified
    # weights (as "occupations") at each k-point. Note, that this automatically puts in the
    # required symmetrization with respect to kpoints and BZ symmetry
    compute_density(basis, ψ, weights; occupation_threshold=weight_threshold)
end
function compute_ldos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
    compute_ldos(ε, scfres.basis, scfres.eigenvalues, scfres.ψ; kwargs...)
end


"""
Projected density of states, onto the atomic orbitals of the given atoms.
"""
function compute_pdos(ε, basis, eigenvalues, ψ; 
                      smearing=basis.model.smearing, 
                      temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_pdos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    guess = DFTK.guess_amn_psp(basis)
    A = DFTK.compute_amn(basis, ψ, guess; spin=1)
    n_projectors = size(A[1], 2)
    PDOS = zeros(eltype(ε), basis.model.n_spin_components, n_projectors)

    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        A = DFTK.compute_amn(basis, ψ, guess; spin=σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature

            # Contribution for each projector
            for iproj in 1:n_projectors
                weight = abs2(A[ik][iband, iproj])
                PDOS[σ, iproj] -= weight * (filled_occ * basis.kweights[ik] / temperature 
                                            * Smearing.occupation_derivative(smearing, enred))
            end
        end
    end
    PDOS = mpi_sum(PDOS, basis.comm_kpts)
end

function compute_pdos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
        compute_pdos(ε, scfres.basis, scfres.eigenvalues, scfres.ψ; kwargs...)
end

# function compute_projected_dos(
#     ε,
#     basis::PlaneWaveBasis, 
#     ψ::AbstractVector{<:AbstractMatrix{<:Complex}}, 
#     eigenvalues::AbstractVector{<:AbstractVector{<:Real}}, 
#     energy_grid::AbstractVector, 
#     smearing=basis.model.smearing,
#     temperature=basis.model.temperature,
#     spin::Integer=1
# )
#     if (temperature == 0) || smearing isa Smearing.None
#         error("compute_projected_dos only supports finite temperature")
#     end
#     filled_occ = filled_occupation(basis.model)

#     # Compute the projection matrix A
#     guess = DFTK.guess_amn_psp(basis)
#     A = DFTK.compute_amn(basis, ψ, guess; spin)

#     # Compute the projected DOS
#     dos_projected = zeros(Float64, length(energy_grid), length(A[1][1, :]))
#     for (ik, kpt) in enumerate(krange_spin(basis, spin))
#         weights_k = A[ik]' * A[ik]  # Weights of the projections at this k-point
#         for (n, εnk) in enumerate(eigenvalues[ik])
#             # Calculate the occupation derivative
#             enred = (εnk - ε) / temperature
#             occ_deriv = -filled_occ / temperature * Smearing.occupation_derivative(smearing, enred)

#             # Find the index of the energy grid closest to this eigenvalue
#             idx = argmin(abs.(energy_grid .- εnk))
#             dos_projected[idx, :] .+= weights_k[n, :] * occ_deriv
#         end
#     end
#     dos_projected ./= length(krange_spin(basis, spin))  # Normaer of klize by numb-points

#     dos_projected
# end



"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_pdos end