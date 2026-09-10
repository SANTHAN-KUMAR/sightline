// Sightline simulation - editor target.

using UnrealBuildTool;
using System.Collections.Generic;

public class SightlineSimEditorTarget : TargetRules
{
	public SightlineSimEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.AddRange(new string[] { "SightlineSim" });
	}
}
