# Test for issue #854: Scalar indexing with multiplication between a scalar and the adjoint of ROC sparse matrices
# https://github.com/JuliaGPU/AMDGPU.jl/issues/854

@testset "Issue #854: Scalar multiplication with adjoint/transpose" begin
    # Test the exact case from the issue
    @testset "Reproducing the original issue" begin
        N = 10
        a = rand(Float64)
        M = ROCSparseMatrixCSC(sprand(ComplexF64, N, N, 0.5))
        
        # This should not trigger scalar indexing
        AMDGPU.allowscalar(false) do
            # Test a * M'
            result = a * M'
            @test result isa Adjoint{ComplexF64, ROCSparseMatrixCSC{ComplexF64, Int32}}
            
            # Test a * transpose(M)
            result = a * transpose(M)
            @test result isa Transpose{ComplexF64, ROCSparseMatrixCSC{ComplexF64, Int32}}
            
            # Test the reverse order
            result = M' * a
            @test result isa Adjoint{ComplexF64, ROCSparseMatrixCSC{ComplexF64, Int32}}
            
            result = transpose(M) * a
            @test result isa Transpose{ComplexF64, ROCSparseMatrixCSC{ComplexF64, Int32}}
        end
        
        # Verify the results match CPU computation
        M_cpu = SparseMatrixCSC(M)
        @test Array(a * M') ≈ a * M_cpu'
        @test Array(M' * a) ≈ M_cpu' * a
        @test Array(a * transpose(M)) ≈ a * transpose(M_cpu)
        @test Array(transpose(M) * a) ≈ transpose(M_cpu) * a
    end
    
    @testset "All sparse matrix types" begin
        N = 10
        a = 2.5
        S = sprand(ComplexF64, N, N, 0.5)
        
        for MatType in [ROCSparseMatrixCSC, ROCSparseMatrixCSR, ROCSparseMatrixCOO]
            M = MatType(S)
            
            AMDGPU.allowscalar(false) do
                # Test all four combinations
                result1 = a * M'
                result2 = M' * a
                result3 = a * transpose(M)
                result4 = transpose(M) * a
                
                @test result1 isa Adjoint
                @test result2 isa Adjoint
                @test result3 isa Transpose
                @test result4 isa Transpose
            end
            
            # Verify correctness
            @test Array(a * M') ≈ a * S'
            @test Array(M' * a) ≈ S' * a
            @test Array(a * transpose(M)) ≈ a * transpose(S)
            @test Array(transpose(M) * a) ≈ transpose(S) * a
        end
        
        # Test BSR separately as it needs blockDim
        M_bsr = ROCSparseMatrixBSR(ROCSparseMatrixCSR(S), 1)
        
        AMDGPU.allowscalar(false) do
            result1 = a * M_bsr'
            result2 = M_bsr' * a
            result3 = a * transpose(M_bsr)
            result4 = transpose(M_bsr) * a
            
            @test result1 isa Adjoint
            @test result2 isa Adjoint
            @test result3 isa Transpose
            @test result4 isa Transpose
        end
        
        @test Array(a * M_bsr') ≈ a * S'
        @test Array(M_bsr' * a) ≈ S' * a
    end
    
    @testset "Different scalar and matrix element types" begin
        N = 10
        S_complex = sprand(ComplexF64, N, N, 0.5)
        S_real = sprand(Float64, N, N, 0.5)
        
        for scalar_type in [Float32, Float64, Int]
            a = scalar_type(2)
            
            # Complex matrix with real/int scalar
            M_complex = ROCSparseMatrixCSC(S_complex)
            AMDGPU.allowscalar(false) do
                result = a * M_complex'
                @test result isa Adjoint
            end
            @test Array(a * M_complex') ≈ a * S_complex'
            
            # Real matrix with real/int scalar
            M_real = ROCSparseMatrixCSC(S_real)
            AMDGPU.allowscalar(false) do
                result = a * M_real'
                @test result isa Adjoint
            end
            @test Array(a * M_real') ≈ a * S_real'
        end
    end
end
