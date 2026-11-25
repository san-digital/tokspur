#pragma once
// Lossless planar weight pack v3 (tokspur-pack-v3).
// Eliminates GGUF scale bit-twiddling and AoS unaligned strides while preserving
// 100 per cent identical FP32 dequantised operands and byte-exact GGUF reconstruction
// with zero sidecar footer bytes.

#include "tokspur/derived_pack.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur::planar_v3 {

inline constexpr uint32_t pack_version_v3 = 3;
inline constexpr const char* pack_recipe_v3 = "tokspur-pack-v3";
inline constexpr const char* block_recipe_prefix_v3 = "tokspur-pack-v3-block-";
inline constexpr uint64_t expected_pack_v3_payload_bytes = 85'881'974'272ULL;
inline constexpr uint64_t max_streaming_scratch_bytes = 96ULL * 1024 * 1024; // 96 MiB (well within 4.00 GB ceiling)

enum class PlanarKind : uint32_t {
    plain = 0,
    q8_lanes_v1 = 1,
    q4k_coloc_gate_v3 = 2,
    q4k_coloc_up_v3 = 3,
    q5k_coloc_gate_v3 = 4,
    q5k_coloc_up_v3 = 5,
    q8_coloc_gate_v3 = 6,
    q8_coloc_up_v3 = 7,
    q51_planar_v3 = 8,
    q8_planar_v3 = 9,
};

[[nodiscard]] const char* planar_kind_name(PlanarKind kind);
[[nodiscard]] PlanarKind parse_planar_kind(std::string_view name);
[[nodiscard]] bool is_planar_v3_kind(PlanarKind kind) noexcept;
[[nodiscard]] bool is_coloc_gate_kind(PlanarKind kind) noexcept;
[[nodiscard]] bool is_coloc_up_kind(PlanarKind kind) noexcept;

struct PlanarExpertSpec {
    uint32_t source_type = 0; // 7 (Q5_1), 8 (Q8_0), 12 (Q4_K), 13 (Q5_K)
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint32_t blocks_per_row = 0;
    uint64_t source_row_bytes = 0;
    uint64_t planar_row_bytes = 0;
    uint64_t source_expert_bytes = 0;
    uint64_t planar_expert_bytes = 0;
    uint64_t qs_offset = 0;
    uint64_t qh_offset = 0;
    uint64_t sm_offset = 0;
    uint64_t dm_offset = 0;
};

[[nodiscard]] PlanarExpertSpec make_expert_spec(uint32_t source_type, uint32_t columns, uint32_t rows);

// Zero-dependency compile-time / runtime shape & tiling invariant discharge for Metal planar v3 kernels.
// Proves that decode16_planar_v3, decode32_planar_v3, and stage64_planar_gu_v3 block indices never exceed blocks_per_row.
[[nodiscard]] constexpr bool discharge_planar_tiling_invariants(
    uint32_t source_type, uint32_t columns, uint32_t rows, uint32_t experts) noexcept {
    if (experts < 1u || experts > 512u || rows < 4u || rows > 65536u || (rows % 4u) != 0u) return false;
    if (columns < 32u || columns > 65536u) return false;
    const uint32_t block_cols = (source_type == 12u || source_type == 13u) ? 256u
                              : ((source_type == 7u || source_type == 8u) ? 32u : 0u);
    if (block_cols == 0u || (columns % block_cols) != 0u) return false;
    const uint32_t blocks_per_row = columns / block_cols;
    const uint32_t steps = columns / 32u;
    if (blocks_per_row == 0u || steps == 0u) return false;
    if (source_type == 8u && ((uint64_t(rows) * blocks_per_row) % 8ull) != 0ull) return false;
    // decode16_planar_v3: ci in [0, 2*steps - 1]
    const uint32_t max_ci = 2u * steps - 1u;
    const uint32_t max_blk16 = (source_type == 12u || source_type == 13u) ? (max_ci >> 4u) : (max_ci >> 1u);
    if (max_blk16 >= blocks_per_row) return false;
    // decode32_planar_v3: group32 in [0, steps - 1]
    const uint32_t max_g32 = steps - 1u;
    const uint32_t max_blk32 = (source_type == 12u || source_type == 13u) ? (max_g32 >> 3u) : max_g32;
    if (max_blk32 >= blocks_per_row) return false;
    // stage64_planar_gu_v3 (Q4_K / Q5_K): window64 in [0, columns/64 - 1]
    if (source_type == 12u || source_type == 13u) {
        if ((columns % 64u) != 0u) return false;
        const uint32_t max_sb64 = (columns / 64u - 1u) >> 2u;
        if (max_sb64 >= blocks_per_row) return false;
    }
    return true;
}

static_assert(discharge_planar_tiling_invariants(12u, 256u, 4u, 2u));
static_assert(discharge_planar_tiling_invariants(13u, 2560u, 640u, 512u));
static_assert(discharge_planar_tiling_invariants(12u, 65536u, 65536u, 512u));
static_assert(discharge_planar_tiling_invariants(7u, 96u, 512u, 1u));
static_assert(discharge_planar_tiling_invariants(8u, 640u, 2560u, 512u));
static_assert(!discharge_planar_tiling_invariants(12u, 128u, 64u, 2u));
static_assert(!discharge_planar_tiling_invariants(12u, 256u, 6u, 2u));
static_assert(!discharge_planar_tiling_invariants(8u, 32u, 4u, 1u));
static_assert(!discharge_planar_tiling_invariants(7u, 64u, 4u, 513u));

// Statically certify finite dm (d, dmin), 6-bit sm <= 63, and FP16-staging safety (|wf| < 65520.0f).
void certify_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert);
void certify_planar_matrix_bytes(const PlanarExpertSpec& spec, uint32_t experts, std::span<const std::byte> matrix_bytes);
struct CertifiedPlanarTensor {
    uint64_t pack_data_offset = 0;
    uint64_t bytes = 0;
    uint32_t source_type = 0;
    uint32_t kind = 0;
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint32_t experts = 0;
    bool operator==(const CertifiedPlanarTensor&) const = default;
};
enum class CertifiedTensorLookup : uint8_t {
    not_certified_model = 0,
    exact_match = 1,
    not_found = 2,
    mismatched_overlap = 3,
};
[[nodiscard]] bool is_certified_pack_model(const std::shared_ptr<const ModelFiles>& model) noexcept;
[[nodiscard]] CertifiedTensorLookup lookup_certified_planar_tensor(
    const std::shared_ptr<const ModelFiles>& model,
    const CertifiedPlanarTensor& query) noexcept;

// Bijective 12-byte GGUF Q4_K/Q5_K scales[12] <-> 16-byte {uint8 sc[8], uint8 mn[8]} in [0, 63].
void pack_sm16(std::span<const std::byte, 12> g_scales, std::span<std::byte, 16> out_sm) noexcept;
void unpack_sm16(std::span<const std::byte, 16> in_sm, std::span<std::byte, 12> out_scales);

// Pack / unpack one complete expert matrix ([columns, rows]) between GGUF AoS and planar v3 SoA.
void pack_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> gguf_expert, std::span<std::byte> planar_expert);
void unpack_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<std::byte> gguf_expert);

// Exact CPU dequantisation of one planar v3 expert into FP32 (matching tokspur::dequantize and GPU FP32 decode).
void dequantize_expert_planar_v3(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<float> output);
void dequantize_expert_planar_v3_f32(const PlanarExpertSpec& spec, std::span<const std::byte> planar_expert, std::span<float> output);

struct PackTensorV3 {
    std::string name;
    uint32_t type = 0;               // GGUF directory type (24 = I8 when bytes != source_bytes, else source_type)
    uint32_t source_type = 0;        // Original GGUF encoding
    PlanarKind kind = PlanarKind::plain;
    std::string layout = "plain";
    std::vector<uint64_t> dimensions;        // GGUF directory dimensions
    std::vector<uint64_t> source_dimensions; // Original GGUF dimensions
    uint64_t source_bytes = 0;
    uint64_t bytes = 0;
    uint64_t offset = 0; // Relative to data_offset, 16 KiB aligned
    std::string partner_name; // Non-empty for co-located gate/up pairs
    std::string source_sha256;
    std::string packed_sha256;
};

struct PackPlanV3 {
    std::vector<PackTensorV3> tensors;
    uint64_t source_payload_bytes = 0;
    uint64_t payload_bytes = 0;
    uint64_t file_bytes = 0;
    uint64_t data_offset = 0;
    uint64_t q4k_superblocks = 0;
    uint64_t q5k_superblocks = 0;
    uint64_t original_12b_scale_header_bytes = 0;
    uint64_t expanded_16b_scale_header_bytes = 0;
    std::string recipe = pack_recipe_v3;
    std::string target_revision = derived_pack::target_revision;
    uint32_t version = pack_version_v3;
    uint32_t alignment = static_cast<uint32_t>(derived_pack::pack_alignment);
};

struct BudgetReportV3 {
    uint64_t filesystem_available_bytes = 0;
    uint64_t existing_pack_v2_file_bytes = 0;
    uint64_t planned_pack_v3_payload_bytes = 0;
    uint64_t planned_pack_v3_file_bytes = 0;
    uint64_t free_after_pack_v3_bytes = 0;
    uint64_t required_reserve_bytes = derived_pack::default_reserve_bytes;
    uint64_t streaming_ring_buffer_bytes = 0;
    uint64_t streaming_ram_ceiling_bytes = 4'000'000'000ULL;
    uint64_t q4k_planar_bytes = 0;
    uint64_t q5k_planar_bytes = 0;
    uint64_t q51_planar_bytes = 0;
    uint64_t q8_planar_moe_bytes = 0;
    uint64_t original_12b_scale_header_bytes = 0;
    uint64_t expanded_16b_scale_header_bytes = 0;
    bool reserve_satisfied = false;
    bool ram_ceiling_satisfied = false;
};

struct AdoptedPackV3 {
    std::shared_ptr<const ModelFiles> model;
    PackPlanV3 plan;
    std::string manifest_sha256;
    uint32_t authentication_workers = 1;
    // Describes completed source reads, not pages released by the OS.
    std::string_view authentication_source_read_mode = "not_read";
    derived_pack::LocalSealUse local_seal = derived_pack::LocalSealUse::not_consulted;
    derived_pack::PackFileIdentity pack_identity; // of the descriptor that was adopted, after any verification
    derived_pack::SealScope compared;             // what this call fully verified the pack against; empty when the seal was accepted
};

[[nodiscard]] std::vector<std::byte> build_header_v3(const PackPlanV3& plan);

// Full 1,254-tensor tokspur-pack-v3 planner and disk/RAM budget inspector.
[[nodiscard]] PackPlanV3 plan_derived_pack_v3(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar = nullptr,
    uint64_t byte_budget = derived_pack::max_pack_bytes
);

[[nodiscard]] BudgetReportV3 measure_pack_v3_budget(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar,
    const std::filesystem::path& output_directory,
    const std::filesystem::path& existing_pack_v2_path = {},
    uint32_t experts_per_chunk = 8,
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes
);

[[nodiscard]] BudgetReportV3 measure_pack_v3_budget(
    const PackPlanV3& plan, const std::filesystem::path& output_directory,
    const std::filesystem::path& existing_pack_v2_path, uint32_t experts_per_chunk,
    uint64_t reserve_bytes, uint64_t chunk_bytes
);

struct PublishedPackV3 {
    AdoptedPackV3 verified;
    bool existed;
    double build_seconds;
};

// Fully authenticate against the supplied sources before publishing or accepting a seal.
// The seal (see derived_pack::TrustMode) records the manifest hash and the identity of
// the verified pack file, with the scope target+mtp since both sources are required
// here. A missing seal is created. An existing seal that already vouches for the pack
// is kept. A sound seal for the same manifest that is not current (the earlier
// manifest-only format, another file identity, or a narrower scope) is replaced by rename.
// Any other existing seal is refused and left untouched.
[[nodiscard]] PublishedPackV3 build_or_verify_pack_v3(
    PackPlanV3 plan, std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar, const std::filesystem::path& output,
    uint32_t experts_per_chunk, uint64_t reserve_bytes, uint64_t chunk_bytes,
    void (*check_cancelled)() = nullptr
);

void validate_pack_v3_output_name(const std::filesystem::path& output);
void require_full_pack_v3_census(const PackPlanV3& plan);
int pack_v3_cli(int argc, char** argv);

// Single-block fixture planner (e.g. block_id in {0, 2, 3, 48}) for sequential D1 prototype.
[[nodiscard]] PackPlanV3 plan_block_pack_v3(
    const ModelFiles& source,
    const ModelFiles* mtp_sidecar,
    uint32_t block_id,
    uint64_t byte_budget = 4'000'000'000ULL
);

[[nodiscard]] PackPlanV3 build_pack_v3_from_plan(
    PackPlanV3 plan,
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk = 8,
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes,
    void (*check_cancelled)() = nullptr
);

[[nodiscard]] PackPlanV3 build_derived_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk = 8,
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes
);

[[nodiscard]] PackPlanV3 build_block_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    uint32_t block_id,
    const std::filesystem::path& output,
    uint32_t experts_per_chunk = 8,
    uint64_t reserve_bytes = derived_pack::default_reserve_bytes,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes
);

[[nodiscard]] AdoptedPackV3 adopt_pack_v3_with_plan(
    PackPlanV3 expected_plan,
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk = 8,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes,
    uint32_t authentication_workers = 1,
    derived_pack::TrustMode trust_mode = derived_pack::TrustMode::full,
    bool require_full_census = true,
    void (*check_cancelled)() = nullptr
);

[[nodiscard]] AdoptedPackV3 adopt_derived_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk = 8,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes,
    uint32_t authentication_workers = 1,
    derived_pack::TrustMode trust_mode = derived_pack::TrustMode::full,
    bool require_full_census = true
);

[[nodiscard]] AdoptedPackV3 adopt_block_pack_v3(
    std::shared_ptr<const ModelFiles> source,
    std::shared_ptr<const ModelFiles> mtp_sidecar,
    uint32_t block_id,
    const std::filesystem::path& path,
    uint32_t experts_per_chunk = 8,
    uint64_t chunk_bytes = derived_pack::default_chunk_bytes,
    uint32_t authentication_workers = 1
);

} // namespace tokspur::planar_v3
