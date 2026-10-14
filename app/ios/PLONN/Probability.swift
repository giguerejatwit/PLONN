import Foundation

struct ProbabilityLeg {
    let event: String
    let probability: Double?
    let mode: String
}

func fair(_ p: Double?) -> Double? { guard let p, p > 0, p < 1 else { return nil }; return p >= 0.5 ? -100*p/(1-p) : 100*(1-p)/p }
func implied(_ price: Double) -> Double { price > 0 ? 100/(price+100) : -price/(-price+100) }

func parlayEstimate(_ legs: [ProbabilityLeg]) -> (Double?, String) {
    guard legs.count >= 2 else { return (nil, "Add at least two selections.") }
    guard Set(legs.map(\.mode)).count == 1 else { return (nil, "Sample and live selections cannot be combined.") }
    guard legs.allSatisfy({ !$0.event.isEmpty }), Set(legs.map(\.event)).count == legs.count else {
        return (nil, "Same-game legs require a validated joint model. Probability unavailable.")
    }
    guard legs.allSatisfy({ guard let p = $0.probability else { return false }; return p.isFinite && p > 0 && p < 1 }) else {
        return (nil, "A validated model estimate is unavailable for one or more legs.")
    }
    return (legs.reduce(1) { $0 * ($1.probability ?? 0) }, "Independence assumption. Different events may still have dependencies.")
}

struct PropDistribution {
 let values:[Double]
 let line:Double
 var mean:Double { values.reduce(0,+)/Double(values.count) }
 var median:Double { let v=values.sorted();let i=v.count/2;return v.count % 2 == 0 ? (v[i-1]+v[i])/2:v[i] }
 var variance:Double { values.reduce(0){$0+pow($1-mean,2)}/Double(values.count-1) }
 var deviation:Double { sqrt(variance) }
 init?(values:[Double],line:Double) {
  guard values.count >= 5,values.allSatisfy({$0.isFinite && $0 >= 0 && $0 <= 10000}),line.isFinite,line >= 0,line <= 10000 else{return nil}
  self.values=values;self.line=line
 }
 var empirical:(over:Double,under:Double,push:Double) {
  let n=Double(values.count)
  return (Double(values.filter{$0>line}.count)/n,Double(values.filter{$0<line}.count)/n,Double(values.filter{$0==line}.count)/n)
 }
 func interval(under:Bool)->(Double,Double) {
  let p=under ? empirical.under:empirical.over,n=Double(values.count),z=1.96
  let denominator=1+z*z/n,center=(p+z*z/(2*n))/denominator
  let width=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
  return (max(0,center-width),min(1,center+width))
 }
 var poisson:(over:Double,under:Double,push:Double)? {
  guard values.allSatisfy({$0.rounded()==$0}),mean <= 500 else{return nil}
  var mass=exp(-mean),cdf=mass
  if Int(line)>=1 { for k in 1...Int(line){mass *= mean/Double(k);cdf += mass} }
  let push=line.rounded()==line ? mass:0
  return (max(0,min(1,1-cdf)),max(0,min(1,cdf-push)),push)
 }
 var normal:(over:Double,under:Double,push:Double)? {
  guard deviation>0 else{return nil}
  func cdf(_ x:Double)->Double { 0.5*erfc(-(x-mean)/(deviation*sqrt(2))) }
  let counts=values.allSatisfy{$0.rounded()==$0}
  if counts {
   let under=cdf(ceil(line)-0.5),over=1-cdf(floor(line)+0.5)
   return (over,under,max(0,1-over-under))
  }
  return (1-cdf(line),cdf(line),0)
 }
}
