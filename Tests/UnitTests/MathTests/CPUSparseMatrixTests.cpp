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

BOOST_FIXTURE_TEST_CASE(CPUSparseMatrixAdd, RandomSeedFixture)
{
    const size_t m = 4;
    const size_t n = 2;
    
    double data0[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    DenseMatrix dm0(m, n, data0);

    double data1[] = {0, 0, 0, 4, 0, 0, 5, 6};
    DenseMatrix dm1(m, n, data1);
    double val[] = {4, 5, 6};
    int cscRow[] = {3, 2, 3};
    int cscCol[] = {0, 1, 3};
    SparseMatrix sm1(MatrixFormat::matrixFormatSparseCSC, m, n, 0);
    sm1.SetMatrixFromCSCFormat(cscCol, cscRow, val, _countof(val), 4, 2);
    foreach_coord(row, col, dm1)
    {
        BOOST_CHECK(sm1(row, col) == dm1(row, col));
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
    SparseMatrix::MultiplyAndAdd(1, dm1, false, sm1, true, smMul2);

    DenseMatrix dmMul2(m, m);
    DenseMatrix::MultiplyAndAdd(dm1, false, dm1, true, dmMul2);
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

    double data3[] = { 0, 0, 0, 4, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0 };
    DenseMatrix dm3(m, m, data3);
    double nzBlocks[] = { 0, 0, 0, 4, 0, 0, 5, 6 };
    size_t blockIds[] = { 0, 1 };
    SparseMatrix sm3(MatrixFormat::matrixFormatSparseBlockCol);
    sm3.SetMatrixFromSBCFormat(blockIds, nzBlocks, _countof(blockIds), m, m);
    foreach_coord(row, col, dm3)
    {
        BOOST_CHECK(sm3(row, col) == dm3(row, col));
    }

    dmMul2 = (dmMul2 * 0.9) + dm3;
    SparseMatrix::ScaleAndAccumulate(0.9, smMul2, sm3);

    foreach_coord(row, col, dmMul2)
    {
        BOOST_CHECK(smMul2(row, col) == dmMul2(row, col));
    }
}

BOOST_AUTO_TEST_SUITE_END()
}
} } }
