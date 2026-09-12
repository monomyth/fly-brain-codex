import Foundation
import RobotCore
let values = try JSONDecoder().decode([[Double]].self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
let floor = try FloorConstraint(Kinematics(RobotDefinition.load()))
var heights: [Double] = []
var invalid: [Int] = []
for (index, value) in values.enumerated() {
    let height = floor.minimumHeight(Pose(name: "Training target", joints: Array(value.prefix(6)), grip: value[6]))
    heights.append(height * 1000)
    if height < FloorConstraint.height { invalid.append(index) }
}
let result: [String: Any] = ["rows": values.count, "invalid_rows": invalid, "minimum_height_mm": heights.min() ?? 0, "floor_height_mm": FloorConstraint.height * 1000]
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(data)
