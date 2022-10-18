const NUM_LANES: usize = 8;
use super::LowerAsciiCharset;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::mem::transmute;

pub fn filter_vec_avx2(input: &[LowerAsciiCharset], charset: LowerAsciiCharset, last_added: LowerAsciiCharset) -> Vec<LowerAsciiCharset> {
    let mut output = Vec::with_capacity(input.len());
    let num_words = input.len() / NUM_LANES;
    unsafe {
        let output_len = filter_vec_avx2_aux(
            input.as_ptr() as *const __m256i,
            charset,
            last_added,
            output.as_mut_ptr(),
            num_words,
        );
        output.set_len(output_len);
    }

    // don't forget the excess
    for i in 0..(input.len() % NUM_LANES) {
        let idx = num_words * NUM_LANES + i;
        let c = input[idx];
        if !charset.intersects(c) && c > last_added {
            output.push(c);
        }
    }

    output
}

unsafe fn filter_vec_avx2_aux(
    mut input: *const __m256i,
    charset: LowerAsciiCharset,
    last_added: LowerAsciiCharset,
    output: *mut LowerAsciiCharset,
    num_words: usize,
) -> usize {
    let mut output_tail = output;
    let charset_simd = _mm256_set1_epi32(transmute(charset));
    let last_added_simd = _mm256_set1_epi32(transmute(last_added));
    for _ in 0..num_words {
        let word = _mm256_loadu_si256(input);
        let keeper_bitset = compute_filter_bitset(word, charset_simd, last_added_simd);
        let added_len = keeper_bitset.count_ones();
        let compacted_output = compact(word, keeper_bitset);
        _mm256_storeu_si256(output_tail as *mut __m256i, compacted_output);
        output_tail = output_tail.offset(added_len as isize);
        input = input.offset(1);
    }
    output_tail.offset_from(output) as usize
}

#[inline]
unsafe fn compact(data: __m256i, mask: u8) -> __m256i {
    let vperm_mask = BITSET_TO_MAPPING[mask as usize];
    _mm256_permutevar8x32_epi32(data, vperm_mask)
}

#[inline]
unsafe fn compute_filter_bitset(val: __m256i, charset_simd: __m256i, last_added_simd: __m256i) -> u8 {
    let prod: __m256i = _mm256_or_si256(
        _mm256_and_si256(val, charset_simd),
        _mm256_cmpgt_epi32(last_added_simd, val),
    );
    let ztest = transmute::<__m256i, __m256>(_mm256_cmpeq_epi32(prod, _mm256_set1_epi32(0)));
    _mm256_movemask_ps(ztest) as u8
}

const fn from_u32x8(vals: [u32; NUM_LANES]) -> __m256i {
    union U8x32 {
        vector: __m256i,
        vals: [u32; NUM_LANES],
    }
    unsafe { U8x32 { vals }.vector }
}

const BITSET_TO_MAPPING: [__m256i; 256] = [
    from_u32x8([0, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 0, 0, 0, 0, 0, 0]),
    from_u32x8([2, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 0, 0, 0, 0, 0]),
    from_u32x8([3, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 0, 0, 0, 0, 0]),
    from_u32x8([2, 3, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 0, 0, 0, 0]),
    from_u32x8([4, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 4, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 0, 0, 0, 0, 0]),
    from_u32x8([2, 4, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 4, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 0, 0, 0, 0]),
    from_u32x8([3, 4, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 4, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 0, 0, 0, 0]),
    from_u32x8([2, 3, 4, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 4, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 0, 0, 0]),
    from_u32x8([5, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 5, 0, 0, 0, 0, 0]),
    from_u32x8([2, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 5, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 5, 0, 0, 0, 0]),
    from_u32x8([3, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 5, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 5, 0, 0, 0, 0]),
    from_u32x8([2, 3, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 5, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 5, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 5, 0, 0, 0]),
    from_u32x8([4, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 5, 0, 0, 0, 0, 0]),
    from_u32x8([1, 4, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 5, 0, 0, 0, 0]),
    from_u32x8([2, 4, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 5, 0, 0, 0, 0]),
    from_u32x8([1, 2, 4, 5, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 5, 0, 0, 0]),
    from_u32x8([3, 4, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 5, 0, 0, 0, 0]),
    from_u32x8([1, 3, 4, 5, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 5, 0, 0, 0]),
    from_u32x8([2, 3, 4, 5, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 5, 0, 0, 0]),
    from_u32x8([1, 2, 3, 4, 5, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 5, 0, 0]),
    from_u32x8([6, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 6, 0, 0, 0, 0, 0]),
    from_u32x8([2, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 6, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 6, 0, 0, 0, 0]),
    from_u32x8([3, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 6, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 6, 0, 0, 0, 0]),
    from_u32x8([2, 3, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 6, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 6, 0, 0, 0]),
    from_u32x8([4, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 6, 0, 0, 0, 0, 0]),
    from_u32x8([1, 4, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 6, 0, 0, 0, 0]),
    from_u32x8([2, 4, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 6, 0, 0, 0, 0]),
    from_u32x8([1, 2, 4, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 6, 0, 0, 0]),
    from_u32x8([3, 4, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 6, 0, 0, 0, 0]),
    from_u32x8([1, 3, 4, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 6, 0, 0, 0]),
    from_u32x8([2, 3, 4, 6, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 6, 0, 0, 0]),
    from_u32x8([1, 2, 3, 4, 6, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 6, 0, 0]),
    from_u32x8([5, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([1, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 5, 6, 0, 0, 0, 0]),
    from_u32x8([2, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 5, 6, 0, 0, 0, 0]),
    from_u32x8([1, 2, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 5, 6, 0, 0, 0]),
    from_u32x8([3, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 5, 6, 0, 0, 0, 0]),
    from_u32x8([1, 3, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 5, 6, 0, 0, 0]),
    from_u32x8([2, 3, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 5, 6, 0, 0, 0]),
    from_u32x8([1, 2, 3, 5, 6, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 5, 6, 0, 0]),
    from_u32x8([4, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 5, 6, 0, 0, 0, 0]),
    from_u32x8([1, 4, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 5, 6, 0, 0, 0]),
    from_u32x8([2, 4, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 5, 6, 0, 0, 0]),
    from_u32x8([1, 2, 4, 5, 6, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 5, 6, 0, 0]),
    from_u32x8([3, 4, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 5, 6, 0, 0, 0]),
    from_u32x8([1, 3, 4, 5, 6, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 5, 6, 0, 0]),
    from_u32x8([2, 3, 4, 5, 6, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 5, 6, 0, 0]),
    from_u32x8([1, 2, 3, 4, 5, 6, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 5, 6, 0]),
    from_u32x8([7, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 7, 0, 0, 0, 0, 0]),
    from_u32x8([2, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 7, 0, 0, 0, 0]),
    from_u32x8([3, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 7, 0, 0, 0, 0]),
    from_u32x8([2, 3, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 7, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 7, 0, 0, 0]),
    from_u32x8([4, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 4, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 7, 0, 0, 0, 0]),
    from_u32x8([2, 4, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 7, 0, 0, 0, 0]),
    from_u32x8([1, 2, 4, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 7, 0, 0, 0]),
    from_u32x8([3, 4, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 7, 0, 0, 0, 0]),
    from_u32x8([1, 3, 4, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 7, 0, 0, 0]),
    from_u32x8([2, 3, 4, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 7, 0, 0, 0]),
    from_u32x8([1, 2, 3, 4, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 7, 0, 0]),
    from_u32x8([5, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 5, 7, 0, 0, 0, 0]),
    from_u32x8([2, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 5, 7, 0, 0, 0, 0]),
    from_u32x8([1, 2, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 5, 7, 0, 0, 0]),
    from_u32x8([3, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 5, 7, 0, 0, 0, 0]),
    from_u32x8([1, 3, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 5, 7, 0, 0, 0]),
    from_u32x8([2, 3, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 5, 7, 0, 0, 0]),
    from_u32x8([1, 2, 3, 5, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 5, 7, 0, 0]),
    from_u32x8([4, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 5, 7, 0, 0, 0, 0]),
    from_u32x8([1, 4, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 5, 7, 0, 0, 0]),
    from_u32x8([2, 4, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 5, 7, 0, 0, 0]),
    from_u32x8([1, 2, 4, 5, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 5, 7, 0, 0]),
    from_u32x8([3, 4, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 5, 7, 0, 0, 0]),
    from_u32x8([1, 3, 4, 5, 7, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 5, 7, 0, 0]),
    from_u32x8([2, 3, 4, 5, 7, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 5, 7, 0, 0]),
    from_u32x8([1, 2, 3, 4, 5, 7, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 5, 7, 0]),
    from_u32x8([6, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 6, 7, 0, 0, 0, 0]),
    from_u32x8([2, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 6, 7, 0, 0, 0, 0]),
    from_u32x8([1, 2, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 6, 7, 0, 0, 0]),
    from_u32x8([3, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 6, 7, 0, 0, 0, 0]),
    from_u32x8([1, 3, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 6, 7, 0, 0, 0]),
    from_u32x8([2, 3, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 6, 7, 0, 0, 0]),
    from_u32x8([1, 2, 3, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 6, 7, 0, 0]),
    from_u32x8([4, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 6, 7, 0, 0, 0, 0]),
    from_u32x8([1, 4, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 6, 7, 0, 0, 0]),
    from_u32x8([2, 4, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 6, 7, 0, 0, 0]),
    from_u32x8([1, 2, 4, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 6, 7, 0, 0]),
    from_u32x8([3, 4, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 6, 7, 0, 0, 0]),
    from_u32x8([1, 3, 4, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 6, 7, 0, 0]),
    from_u32x8([2, 3, 4, 6, 7, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 6, 7, 0, 0]),
    from_u32x8([1, 2, 3, 4, 6, 7, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 6, 7, 0]),
    from_u32x8([5, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([1, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 5, 6, 7, 0, 0, 0]),
    from_u32x8([2, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 5, 6, 7, 0, 0, 0]),
    from_u32x8([1, 2, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 5, 6, 7, 0, 0]),
    from_u32x8([3, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 3, 5, 6, 7, 0, 0, 0]),
    from_u32x8([1, 3, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 3, 5, 6, 7, 0, 0]),
    from_u32x8([2, 3, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 2, 3, 5, 6, 7, 0, 0]),
    from_u32x8([1, 2, 3, 5, 6, 7, 0, 0]),
    from_u32x8([0, 1, 2, 3, 5, 6, 7, 0]),
    from_u32x8([4, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 4, 5, 6, 7, 0, 0, 0]),
    from_u32x8([1, 4, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 4, 5, 6, 7, 0, 0]),
    from_u32x8([2, 4, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 2, 4, 5, 6, 7, 0, 0]),
    from_u32x8([1, 2, 4, 5, 6, 7, 0, 0]),
    from_u32x8([0, 1, 2, 4, 5, 6, 7, 0]),
    from_u32x8([3, 4, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 3, 4, 5, 6, 7, 0, 0]),
    from_u32x8([1, 3, 4, 5, 6, 7, 0, 0]),
    from_u32x8([0, 1, 3, 4, 5, 6, 7, 0]),
    from_u32x8([2, 3, 4, 5, 6, 7, 0, 0]),
    from_u32x8([0, 2, 3, 4, 5, 6, 7, 0]),
    from_u32x8([1, 2, 3, 4, 5, 6, 7, 0]),
    from_u32x8([0, 1, 2, 3, 4, 5, 6, 7]),
];
