#pragma once

#include <numeric>

#include "enoki/random.h"
#include "src/types.h"

class Sampler {
public:
  Sampler() = delete;
  explicit Sampler(uint32_t num_samples) : m_sample_cnt{num_samples} {
    auto seed = enoki::arange<UInt64C>(num_samples);
    auto base_seed = PCG32_DEFAULT_STATE;
    auto seed_value = seed + base_seed;
    auto idx = enoki::arange<UInt64C>(seed_value.size());
    m_pcg.seed(sample_tea_64(seed_value, idx), sample_tea_64(idx, seed_value));
  }

  uint32_t sample_cnt() const { return m_sample_cnt; }

  // range: [min, max)
  template <bool ad> Int<ad> next_int(int32_t min, int32_t max) const {
    return Int<ad>{m_pcg.next_uint32_bounded(max - min)} + min;
  }

  template <bool ad>
  Int<ad> next_int(int32_t min, int32_t max, const Mask<ad> &valid) const {
    if constexpr (ad) {
      return IntD{m_pcg.next_uint32_bounded(max - min, enoki::detach(valid))} +
             min;
    } else {
      return IntC{m_pcg.next_uint32_bounded(max - min, valid)} + min;
    }
  }

  template <bool ad> Float<ad> next_float() const {
    return Float<ad>{m_pcg.next_float32()};
  }

  template <bool ad> Float<ad> next_float(const Mask<ad> &valid) const {
    if constexpr (ad) {
      return FloatD{m_pcg.next_float32(enoki::detach(valid))};
    } else {
      return m_pcg.next_float32(valid);
    }
  }

  template <int n, bool ad> inline Vecf<n, ad> next_nd() const {
    static_assert(n > 0);
    if constexpr (n == 1) {
      return next_float<ad>();
    } else {
      constexpr int m = n / 2;
      return concat(next_nd<m, ad>(), next_nd<n - m, ad>());
    }
  }

  template <int n, bool ad>
  inline Vecf<n, ad> next_nd(const Mask<ad> &valid) const {
    static_assert(n > 0);
    if constexpr (n == 1) {
      return next_float<ad>(valid);
    } else {
      constexpr int m = n / 2;
      return concat(next_nd<m, ad>(valid), next_nd<n - m, ad>(valid));
    }
  }

private:
  enoki::uint64_array_t<UIntC> sample_tea_64(UIntC v0, UIntC v1,
                                             int rounds = 4) {
    UIntC sum = 0;
    for (int i = 0; i < rounds; ++i) {
      sum += 0x9e3779b9;
      v0 += (sl<4>(v1) + 0xa341316c) ^ (v1 + sum) ^ (sr<5>(v1) + 0xc8013ea4);
      v1 += (sl<4>(v0) + 0xad90777d) ^ (v0 + sum) ^ (sr<5>(v0) + 0x7e95761e);
    }
    return enoki::uint64_array_t<UIntC>(v0) +
           enoki::sl<32>(enoki::uint64_array_t<UIntC>(v1));
  }

  uint32_t m_sample_cnt;
  mutable enoki::PCG32<UIntC> m_pcg;
};

class DiscreteSampler {
public:
  DiscreteSampler() = delete;
  explicit DiscreteSampler(const FloatD &pmf_unnorm)
      : DiscreteSampler{enoki::detach(pmf_unnorm)} {}
  explicit DiscreteSampler(const FloatC &pmf_unnorm) {
    m_cnt = enoki::slices(pmf_unnorm);
    m_sum = enoki::hsum(pmf_unnorm);
    m_pmf = pmf_unnorm / m_sum;
    m_cmf = enoki::psum(m_pmf);
  }

  template <bool ad>
  std::tuple<Int<ad>, Float<ad>>
  sample_reuse(Float<ad> &sample, const Mask<ad> &valid = true) const {
    if (m_cnt == 1) {
      return {0, 1};
    }
    auto idx = enoki::binary_search(
        0, m_cnt - 1, [&sample, &valid, this](const Int<ad> &i) {
          return enoki::gather<Float<ad>>(Float<ad>{this->m_cmf}, i, valid) <
                 sample;
        });
    auto pdf = enoki::gather<Float<ad>>(Float<ad>{m_pmf}, idx, valid);

    sample =
        (sample -
         enoki::select(
             idx > 0, enoki::gather<Float<ad>>(Float<ad>{m_cmf}, idx - 1), 0)) /
        pdf;
    return {idx, pdf};
  }

private:
  int32_t m_cnt;
  FloatC m_pmf;
  FloatC m_sum;
  FloatC m_cmf;
};

using AliasSampler = DiscreteSampler;

// class AliasSampler final {
// public:
//   AliasSampler() = delete;
//   explicit AliasSampler(const FloatD &pmf_)
//       : AliasSampler{enoki::detach(pmf_)} {}
//   explicit AliasSampler(const FloatC &pmf_) {
//     auto n = enoki::slices(pmf_);
//     m_cnt = n;
//     auto sum_ = enoki::hsum(pmf_);
//     m_pmf = pmf_ / sum_;
//     auto pmfxn_ = m_pmf * n;
//     auto small = std::vector<Integer>{};
//     auto large = std::vector<Integer>{};
//     auto pmfxn = std::vector<Real>{};
//     for (auto i = 0u; i < n; ++i) {
//       auto xn = pmfxn_[i];
//       pmfxn.emplace_back(xn);
//       xn < 1 ? small.push_back(i) : large.push_back(i);
//     }
//     auto accept = std::vector<Real>(n, 0.f);
//     auto alias = std::vector<Integer>(n, n + 1);
//     while (!small.empty() and !large.empty()) {
//       auto small_idx = small.back();
//       auto large_idx = large.back();
//       small.pop_back();
//       large.pop_back();
//       accept[small_idx] = pmfxn[small_idx];
//       alias[small_idx] = large_idx;
//       pmfxn[large_idx] = pmfxn[large_idx] - (1 - pmfxn[small_idx]);
//       if (pmfxn[large_idx] < 1) {
//         small.push_back(large_idx);
//       } else {
//         large.push_back(large_idx);
//       }
//     }
//     while (!large.empty()) {
//       auto large_idx = large.back();
//       large.pop_back();
//       accept[large_idx] = 1;
//     }
//     while (!small.empty()) {
//       auto small_idx = small.back();
//       small.pop_back();
//       accept[small_idx] = 1;
//     }

//     m_accept = FloatC::copy(accept.data(), n);
//     m_alias = IntC::copy(alias.data(), n);
//   }

//   template <bool ad>
//   std::tuple<Int<ad>, Float<ad>>
//   sample_reuse(Float<ad> &sampler, const Mask<ad> &valid = true) const {
//     if (m_cnt == 1) {
//       return {0, 1};
//     }
//     sampler *= m_cnt;
//     auto idx = enoki::floor2int<Int<ad>>(sampler);
//     sampler -= idx;
//     auto q = enoki::gather<Float<ad>>(Float<ad>{m_accept}, idx, valid);
//     idx[sampler > q] = enoki::gather<Int<ad>>(Int<ad>{m_alias}, idx, valid);
//     auto pdf = enoki::gather<Float<ad>>(Float<ad>{m_pmf}, idx, valid);
//     sampler = enoki::select(sampler < q, sampler / q, (sampler - q) / (1 -
//     q)); sampler[!valid] = 0; return {idx, pdf};
//   }

// private:
//   uint32_t m_cnt;
//   FloatC m_accept;
//   IntC m_alias;
//   FloatC m_pmf;
// };

template <uint32_t N> class HyperCubeSampler {
public:
  HyperCubeSampler() = delete;
  explicit HyperCubeSampler(const VecC<Integer, N> &resolution) {
    auto cells_cnt = 1;
    for (auto i = 0u; i < N; ++i) {
      cells_cnt *= resolution[i][0];
    }
    auto cell_layers = enoki::empty<VecC<Integer, N>>(cells_cnt);
    auto nums = enoki::arange<IntC>(cells_cnt);
    for (auto i = 0, cnt = cells_cnt; i < N; ++i) {
      cnt /= resolution[i][0];
      cell_layers[i] = (cnt == 1 ? nums : nums / cnt);
      nums -= cell_layers[i] * cnt;
    }

    m_cells = cell_layers;
    m_cells_cnt = cells_cnt;
    m_resolution = resolution;
    m_unit = rcp(VecC<Real, N>(m_resolution));
  }

  uint32_t cells_cnt() const { return m_cells_cnt; }

  const VecC<Integer, N> &cells() const { return m_cells; }

  const VecC<Real, N> &unit() const { return m_unit; }

  template <bool ad> void set_mass(const Float<ad> &mass) {
    if (enoki::slices(mass) != m_cells_cnt) {
      // TODO: throw exception
      std::cerr << "mass size not equals to cells cnt" << std::endl;
    }
    // get pmf and construct discrete_sampler
    m_alias_sampler = std::make_shared<AliasSampler>(mass);
  }

  // after call, samples is used to sample boundary edge point and light point
  template <bool ad> Float<ad> sample_reuse(Vecf<N, ad> &samples) const {
    // use samples[-1] with m_pmf to get correct idx and its pmf
    auto &sample = samples[N - 1];
    auto [idx, pdf] = m_alias_sampler->sample_reuse<ad>(sample);
    // reuse samples to sample things in cell[idx], (idx + samples) * unit ->
    // [0,1]^N
    samples = (samples + enoki::gather<Veci<N, ad>>(m_cells, idx)) * m_unit;
    return pdf;
  }

private:
  int32_t m_cells_cnt;
  VecC<Integer, N> m_resolution;
  VecC<Real, N> m_unit;
  VecC<Integer, N> m_cells;
  std::shared_ptr<AliasSampler> m_alias_sampler;
};

using HyperCubeSampler1 = HyperCubeSampler<1>;
using HyperCubeSampler3 = HyperCubeSampler<3>;