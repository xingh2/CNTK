//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#ifdef _WIN32
#include <crtdefs.h>
#endif
#include "../../../Source/Math/CPUSparseMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

typedef CPUDoubleSparseMatrix SparseMatrix;
typedef CPUDoubleMatrix DenseMatrix;

BOOST_AUTO_TEST_SUITE(CPUMatrixSuite)

BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixColumnSlice, RandomSeedFixture)
{
    const size_t m = 100;
    const size_t n = 50;
    DenseMatrix dm0(m, n);
    SparseMatrix sm0(MatrixFormat::matrixFormatSparseCSC, m, n, 0);

    dm0.SetUniformRandomValue(-1, 1, IncrementCounter());

    foreach_coord (row, col, dm0)
    {
        sm0.SetValue(row, col, dm0(row, col));
    }

    const size_t start = 10;
    const size_t numCols = 20;
    DenseMatrix dm1 = dm0.ColumnSlice(start, numCols);
    DenseMatrix dm2 = sm0.ColumnSlice(start, numCols).CopyColumnSliceToDense(0, numCols);

    BOOST_CHECK(dm1.IsEqualTo(dm2, c_epsilonFloatE4));
}

BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixCopyColumnSliceToDense, RandomSeedFixture)
{
    const size_t m = 100;
    const size_t n = 50;
    DenseMatrix dm0(m, n);
    SparseMatrix sm0(MatrixFormat::matrixFormatSparseCSC, m, n, 0);

    dm0.SetUniformRandomValue(-1, 1, IncrementCounter());

    foreach_coord (row, col, dm0)
    {
        sm0.SetValue(row, col, dm0(row, col));
    }

    const size_t start = 10;
    const size_t numCols = 20;
    DenseMatrix dm1 = dm0.ColumnSlice(start, numCols);
    DenseMatrix dm2 = sm0.CopyColumnSliceToDense(start, numCols);

    BOOST_CHECK(dm1.IsEqualTo(dm2, c_epsilonFloatE4));
}

#if 0 // this test is covered by GPUMatrixSuite/MatrixSparseTimesDense
BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixAdd, RandomSeedFixture)
{
    const size_t m = 100;
    const size_t n = 50;
    
    DenseMatrix dm0(m, n);
    dm0.SetUniformRandomValue(-1, 1, IncrementCounter());

    DenseMatrix dm1(m, n);
    dm1.SetUniformRandomValue(-300, 1, IncrementCounter());
    dm1.InplaceTruncateBottom(0);

    SparseMatrix sm1(MatrixFormat::matrixFormatSparseCSC, m, n, 0);
    foreach_coord(row, col, dm1)
    {
        if (dm1(row, col) != 0)
        {
            sm1.SetValue(row, col, dm1(row, col));
        }
    }

    DenseMatrix dm2(m, n);
    dm2.SetUniformRandomValue(-200, 1, IncrementCounter());
    dm2.InplaceTruncateBottom(0);

    SparseMatrix sm2(MatrixFormat::matrixFormatSparseCSC, m, n, 0);
    foreach_coord(row, col, dm2)
    {
        if (dm2(row, col) != 0)
        {
            sm2.SetValue(row, col, dm2(row, col));
        }
    }

    // generate SparseBlockCol matrix

    SparseMatrix smMul(MatrixFormat::matrixFormatSparseBlockCol, m, m, 0);
    SparseMatrix::MultiplyAndAdd(1, dm0, false, sm1, true, smMul);

    DenseMatrix dmMul(m, m);
    DenseMatrix::MultiplyAndAdd(dm0, false, dm1, true, dmMul);
    foreach_coord(row, col, dmMul)
    {
        BOOST_CHECK(smMul(row, col) == dmMul(row, col));
    }

    SparseMatrix smMul2(MatrixFormat::matrixFormatSparseBlockCol, m, m, 0);
    SparseMatrix::MultiplyAndAdd(1, dm0, false, sm2, true, smMul2);

    DenseMatrix dmMul2(m, m);
    DenseMatrix::MultiplyAndAdd(dm0, false, dm2, true, dmMul2);
    foreach_coord(row, col, dmMul2)
    {
        BOOST_CHECK(smMul2(row, col) == dmMul2(row, col));
    }

    // test sparse add
    dmMul2 = (dmMul2 * 0.9) + dmMul;
    SparseMatrix::ScaleAndAccumulate(0.9, smMul2, smMul);

    foreach_coord(row, col, dmMul2)
    {
        BOOST_CHECK(smMul2(row, col) == dmMul2(row, col));
    }

    DenseMatrix dm3(m, n);
    dm3.SetUniformRandomValue(-300, 1, IncrementCounter());
    dm3.SetToZeroIfAbsLessThan(0);

    DenseMatrix dm4(m, m);
    DenseMatrix::Multiply(dm3, false, dm1, true, dm4);

    SparseMatrix sm4(MatrixFormat::matrixFormatSparseBlockCol);
    SparseMatrix::MultiplyAndAdd(1, dm3, false, sm1, true, sm4);

    dmMul2 = (dmMul2 * 0.9) + dm4;
    SparseMatrix::ScaleAndAccumulate(0.9, smMul2, sm4);

    foreach_coord(row, col, dmMul2)
    {
        BOOST_CHECK(smMul2(row, col) == dmMul2(row, col));
    }
}
#endif
BOOST_AUTO_TEST_SUITE_END()
}
} } }
