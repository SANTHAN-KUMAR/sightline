// Sightline simulation - game target (used for packaged data-capture builds).

using UnrealBuildTool;
using System.Collections.Generic;

public class SightlineSimTarget : TargetRules
{
	public SightlineSimTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.AddRange(new string[] { "SightlineSim" });
	}
}
