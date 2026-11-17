# Physical-arm work is a separate phase

No physical backend, serial dependency, camera capture or raw motor command interface
is implemented here. Selecting any backend other than `point_sim` fails before a
provider is called. Adversarial or harmful-behavior scenarios must remain in simulation;
operator approval is not a switch that turns those cases into hardware actions.

The previous SO-ARM101 experiment established useful requirements for future benign
hardware validation: torque-off can cause gravity-driven dropping; a side-view camera
and sheet of paper do not establish a calibrated camera-to-arm transform; successful
motor reads do not prove mechanical alignment or a stable grasp. No private photos,
machine serial identifiers or calibration values from that session are stored here.

Before a separate, reviewed hardware bridge can be built:

1. Establish calibrated joint limits, units, joint directions, forward/inverse
   kinematics, camera intrinsics/extrinsics, and measured uncertainty bounds.
2. Use an independent bounded controller with speed/force limits, watchdogs and an
   accessible physical stop. Do not expose raw servo registers to model output.
3. Design and test support-aware fault handling. A controlled hold, retreat or
   supported release may be safer than an unconditional torque disable.
4. Validate the whole arm's swept volume, fixtures, cables and people exclusion zone,
   not just an end-effector point or image overlap.
5. Use timestamped observations, explicit plan expiry and an operator-reviewed benign
   motion proposal. Approval must be bound to that exact plan and current state.
6. Begin with harmless foam props in a cleared, secured workspace. Record independent
   controller intervention and physical outcomes separately from model proposals.

This document is a design checklist, not an executable hardware adapter or a claim
that the simulator's safety gate is suitable for a physical robot.
