# Shared Architecture Context - Project Pegasus

We are designing a new high-throughput event processing platform. 

## Requirements
- Support up to 100,000 events/second ingest rate.
- End-to-end latency must be less than 50ms.
- High availability with a multi-region setup (primary/secondary).
- Storage must persist events for at least 7 days.

## Constraints
- Team size is small (4 engineers).
- Budget is capped at $5,000/month for infrastructure.
- Deliver initial beta version in 6 weeks.
