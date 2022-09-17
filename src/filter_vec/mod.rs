mod filter_vec_avx2;

use crate::LowerAsciiCharset;

pub fn filter_vec(
    input: &[LowerAsciiCharset],
    charset: LowerAsciiCharset,
) -> Vec<LowerAsciiCharset> {
    // For some reason, the avx2 version of this is much slower on windows for me, but
    // on linux (via WSL2) it is a bit faster, and it was a fun exercise.
    #[cfg(not(windows))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { filter_vec_avx2(input, charset) };
        }
    }

    filter_vec_scalar(input, charset)
}

pub fn filter_vec_scalar(
    input: &[LowerAsciiCharset],
    charset: LowerAsciiCharset,
) -> Vec<LowerAsciiCharset> {
    input
        .iter()
        .copied()
        .filter(|c| !c.intersects(charset))
        .collect()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn filter_vec_avx2(
    input: &[LowerAsciiCharset],
    charset: LowerAsciiCharset,
) -> Vec<LowerAsciiCharset> {
    filter_vec_avx2::filter_vec_avx2(input, charset)
}
