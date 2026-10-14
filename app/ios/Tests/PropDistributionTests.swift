import XCTest
@testable import PLONNCore
final class PropDistributionTests:XCTestCase {
 func testPushAndFrequency(){
  let s=PropDistribution(values:[0,1,2,3,4],line:2)!
  XCTAssertEqual(s.empirical.over,0.4);XCTAssertEqual(s.empirical.under,0.4);XCTAssertEqual(s.empirical.push,0.2)
  XCTAssertEqual(s.median,2);XCTAssertEqual(s.variance,2.5)
  let interval=s.interval(under:false);XCTAssertLessThan(interval.0,0.4);XCTAssertGreaterThan(interval.1,0.4)
 }
 func testCountModelMass(){
  for line in [0.0,1,1.5,4.5] {
   let s=PropDistribution(values:[0,1,2,3,4],line:line)!
   for p in [s.poisson!,s.normal!] { XCTAssertEqual(p.over+p.under+p.push,1,accuracy:1e-10);XCTAssertGreaterThanOrEqual(p.push,0) }
   if line == 1.5 { XCTAssertEqual(s.normal!.push,0,accuracy:1e-10) }
  }
 }
 func testDegenerateAndInvalid(){
  let zero=PropDistribution(values:[0,0,0,0,0],line:0)!
  XCTAssertNil(zero.normal);XCTAssertEqual(zero.poisson!.push,1)
  XCTAssertNil(PropDistribution(values:[1,2],line:1))
  XCTAssertNil(PropDistribution(values:[1,2,3,4,.nan],line:1))
 }
}
