#pragma once

#include "common/status.h"
#include "fused/fused_api.h"

namespace fused {

// Host-only reference (Phase 2). All pointers must be host-accessible.
common::Status FusedReferenceHost(const FusedParams& p);

float DefaultAbsTolerance();

}  // namespace fused
