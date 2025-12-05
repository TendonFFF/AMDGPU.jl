@testset "LinearAlgebra" begin
    @testset "$f(A)±$h(B) $elty" for elty in (
        Float32, Float64, ComplexF32, ComplexF64,
    ), f in (
        identity, transpose, #= adjoint, =#
    ), h in (
        identity, transpose, #= , adjoint, =#
    )
        # adjoint need the support of broadcast for `conj()` to work with `ROCSparseMatrix`.
        n = 10
        alpha = rand()
        beta = rand()
        A = sprand(elty, n, n, rand())
        B = sprand(elty, n, n, rand())

        dA = ROCSparseMatrixCSR(A)
        dB = ROCSparseMatrixCSR(B)

        C = f(A) + h(B)
        dC = f(dA) + h(dB)
        @test C ≈ collect(dC)

        C = f(A) - h(B)
        dC = f(dA) - h(dB)
        @test C ≈ collect(dC)
    end

    @testset "A±$f(B) $elty" for elty in (
        Float32, Float64, ComplexF32, ComplexF64,
    ), f in (
        ROCSparseMatrixCSR, ROCSparseMatrixCSC, x -> ROCSparseMatrixBSR(x, 1),
    )
        n = 10
        A = sprand(elty, n, n, rand())
        B = sprand(elty, n, n, rand())

        dA = ROCSparseMatrixCSR(A)
        dB = ROCSparseMatrixCSR(B)

        C = A + B
        dC = dA + f(dB)
        @test C ≈ collect(dC)

        C = B + A
        dC = f(dB) + dA
        @test C ≈ collect(dC)

        C = A - B
        dC = dA - f(dB)
        @test C ≈ collect(dC)

        C = B - A
        dC = f(dB) - dA
        @test C ≈ collect(dC)
    end

    @testset "$f(A)*b $elty" for elty in (
        Float32, Float64, ComplexF32, ComplexF64,
    ), f in (
        identity, transpose, adjoint,
    )
        n = 10
        alpha = rand()
        beta = rand()
        A = sprand(elty, n, n, rand())
        b = rand(elty, n)
        c = rand(elty, n)
        alpha = beta = 1.0
        c = zeros(elty, n)

        dA = ROCSparseMatrixCSR(A)
        db = ROCArray(b)
        dc = ROCArray(c)

        # test with empty inputs
        @test Array(dA * AMDGPU.zeros(elty, n, 0)) == zeros(elty, n, 0)

        LinearAlgebra.mul!(c, f(A), b, alpha, beta)
        LinearAlgebra.mul!(dc, f(dA), db, alpha, beta)
        @test c ≈ collect(dc)

        if f in (identity, transpose)
            A = A + transpose(A)
            dA = ROCSparseMatrixCSR(A)

            @assert issymmetric(A)
            LinearAlgebra.mul!(c, f(Symmetric(A)), b, alpha, beta)
            LinearAlgebra.mul!(dc, f(Symmetric(dA)), db, alpha, beta)
            @test c ≈ collect(dc)
        else
            A = A + adjoint(A)
            dA = ROCSparseMatrixCSR(A)

            @assert ishermitian(A)
            LinearAlgebra.mul!(c, f(Hermitian(A)), b, alpha, beta)
            LinearAlgebra.mul!(dc, f(Hermitian(dA)), db, alpha, beta)
            @test c ≈ collect(dc)
        end
    end

    @testset "$f(A)*$h(B) $elty" for elty in (
        Float32, Float64, ComplexF32, ComplexF64,
    ), f in (
        identity, transpose, adjoint,
    ), h in (
        identity, transpose, adjoint,
    )
        n = 10
        alpha = rand()
        beta = rand()
        A = sprand(elty, n, n, rand())
        B = rand(elty, n, n)
        C = rand(elty, n, n)

        dA = ROCSparseMatrixCSR(A)
        dB = ROCArray(B)
        dC = ROCArray(C)

        LinearAlgebra.mul!(C, f(A), h(B), alpha, beta)
        LinearAlgebra.mul!(dC, f(dA), h(dB), alpha, beta)
        @test C ≈ collect(dC)

        A = A + transpose(A)
        dA = ROCSparseMatrixCSR(A)

        @assert issymmetric(A)
        LinearAlgebra.mul!(C, f(A), h(B), alpha, beta)
        LinearAlgebra.mul!(dC, f(dA), h(dB), alpha, beta)
        @test C ≈ collect(dC)
    end

    @testset "issue #1095 ($elty)" for elty in (Float32, Float64, ComplexF32, ComplexF64)
        # Test non-square matrices
        n, m, p = 10, 20, 4
        alpha = rand()
        beta = rand()
        A = sprand(elty, n, m, rand())
        B = rand(elty, m, p)
        C = rand(elty, n, p)

        dA = ROCSparseMatrixCSR(A)
        dB = ROCArray(B)
        dC = ROCArray(C)

        LinearAlgebra.mul!(C, A, B, alpha, beta)
        LinearAlgebra.mul!(dC, dA, dB, alpha, beta)
        @test C ≈ collect(dC)

        LinearAlgebra.mul!(B, transpose(A), C, alpha, beta)
        LinearAlgebra.mul!(dB, transpose(dA), dC, alpha, beta)
        @test B ≈ collect(dB)

        LinearAlgebra.mul!(B, adjoint(A), C, alpha, beta)
        LinearAlgebra.mul!(dB, adjoint(dA), dC, alpha, beta)
        @test B ≈ collect(dB)
    end

    @testset "ROCSparseMatrixCSR($f) $elty" for f in (
        transpose, adjoint,
    ), elty in (
        Float32, ComplexF32,
    )
        S = f(sprand(elty, 10, 10, 0.1))
        @test SparseMatrixCSC(ROCSparseMatrixCSR(S)) ≈ S

        S = sprand(elty, 10, 10, 0.1)
        T = f(ROCSparseMatrixCSR(S))
        @test SparseMatrixCSC(ROCSparseMatrixCSC(T)) ≈ f(S)

        S = sprand(elty, 10, 10, 0.1)
        T = f(ROCSparseMatrixCSC(S))
        @test SparseMatrixCSC(ROCSparseMatrixCSR(T)) ≈ f(S)
    end

    @testset "UniformScaling with $typ($dims)" for typ in (
        ROCSparseMatrixCSR, ROCSparseMatrixCSC,
    ), dims in ((10, 10), (5, 10), (10, 5))
        S = sprand(Float32, dims..., 0.1)
        dA = typ(S)

        @test Array(dA + I) ≈ (S + I)
        @test Array(I + dA) ≈ (I + S)
        @test Array(dA - I) ≈ (S - I)
        @test Array(I - dA) ≈ (I - S)
    end

    @testset "Diagonal with $typ(10, 10)" for typ in (
        ROCSparseMatrixCSR, ROCSparseMatrixCSC,
    )
        S = sprand(Float32, 10, 10, 0.8)
        D = Diagonal(rand(Float32, 10))
        dA = typ(S)
        dD = adapt(ROCArray, D)

        @test Array(dA + dD) ≈ S + D
        @test Array(dD + dA) ≈ D + S
        @test Array(dA - dD) ≈ S - D
        @test Array(dD - dA) ≈ D - S

        @test (dA + dD) isa typ
        @test (dD + dA) isa typ
        @test (dA - dD) isa typ
        @test (dD - dA) isa typ
    end

    @testset "Scalar multiplication with adjoint/transpose ($typ, $elty)" for typ in (
        ROCSparseMatrixCSR, ROCSparseMatrixCSC,
    ), elty in (
        Float32, Float64, ComplexF32, ComplexF64,
    )
        N = 10
        S = sprand(elty, N, N, 0.5)
        dS = typ(S)
        
        # Test scalar types that may differ from matrix eltype
        for scalar_elty in (Float32, Float64)
            a = scalar_elty(rand())
            
            # Test scalar * adjoint(matrix)
            expected = a * S'
            result = a * dS'
            @test result isa Adjoint{<:Any, <:typeof(dS)}
            @test Array(result) ≈ expected
            
            # Test adjoint(matrix) * scalar
            expected = S' * a
            result = dS' * a
            @test result isa Adjoint{<:Any, <:typeof(dS)}
            @test Array(result) ≈ expected
            
            # Test scalar * transpose(matrix)
            expected = a * transpose(S)
            result = a * transpose(dS)
            @test result isa Transpose{<:Any, <:typeof(dS)}
            @test Array(result) ≈ expected
            
            # Test transpose(matrix) * scalar
            expected = transpose(S) * a
            result = transpose(dS) * a
            @test result isa Transpose{<:Any, <:typeof(dS)}
            @test Array(result) ≈ expected
        end
    end

    @testset "Scalar multiplication chained operations ($typ, $elty)" for typ in (
        ROCSparseMatrixCSR, ROCSparseMatrixCSC,
    ), elty in (
        Float64, ComplexF64,
    )
        N = 10
        S = sprand(elty, N, N, 0.5)
        dS = typ(S)
        a = rand()
        
        # Test a * M' * M (from issue #854)
        # This should work without scalar indexing
        result = a * dS'
        @test result isa Adjoint{<:Any, <:typeof(dS)}
        
        # Note: M' * M requires matrix-matrix multiplication which is tested elsewhere
    end
end

@testset "SparseArrays.jl" begin
    @testset "findnz" begin
        n = 35
        A = sprand(n, n, 0.2)
        d_A = ROCSparseMatrixCSC(A)
        @test Array(SparseArrays.getcolptr(d_A)) == SparseArrays.getcolptr(A)

        i, j, v = findnz(A)
        d_i, d_j, d_v = findnz(d_A)
        @test Array(d_i) == i && Array(d_j) == j && Array(d_v) == v

        i = unique(sort(rand(1:n, 10)))
        vals = rand(length(i))
        d_i = ROCArray(i)
        d_vals = ROCArray(vals)
        v = sparsevec(i, vals, n)
        d_v = sparsevec(d_i, d_vals, n)
        @test Array(d_v.iPtr) == v.nzind
        @test Array(d_v.nzVal) == v.nzval
        @test d_v.len == v.n
    end

    @testset "Scalar multiplication with adjoint/transpose COO and BSR" begin
        N = 10
        S = sprand(Float64, N, N, 0.5)
        a = 2.0
        
        # Test ROCSparseMatrixCOO
        dS_coo = ROCSparseMatrixCOO(S)
        result = a * dS_coo'
        @test result isa Adjoint{<:Any, <:ROCSparseMatrixCOO}
        @test Array(result) ≈ a * S'
        
        result = dS_coo' * a
        @test result isa Adjoint{<:Any, <:ROCSparseMatrixCOO}
        @test Array(result) ≈ S' * a
        
        result = a * transpose(dS_coo)
        @test result isa Transpose{<:Any, <:ROCSparseMatrixCOO}
        @test Array(result) ≈ a * transpose(S)
        
        # Test ROCSparseMatrixBSR
        dS_bsr = ROCSparseMatrixBSR(ROCSparseMatrixCSR(S), 1)
        result = a * dS_bsr'
        @test result isa Adjoint{<:Any, <:ROCSparseMatrixBSR}
        @test Array(result) ≈ a * S'
        
        result = dS_bsr' * a
        @test result isa Adjoint{<:Any, <:ROCSparseMatrixBSR}
        @test Array(result) ≈ S' * a
        
        result = a * transpose(dS_bsr)
        @test result isa Transpose{<:Any, <:ROCSparseMatrixBSR}
        @test Array(result) ≈ a * transpose(S)
    end
end
