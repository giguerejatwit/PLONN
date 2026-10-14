import Foundation

// Public client configuration. Database access is enforced by RLS.
enum CloudConfiguration {
 static let servicePolicy = PLONNServicePolicy(info: Bundle.main.infoDictionary ?? [:])
 static let nbaPreseasonCacheEnabled = (Bundle.main.infoDictionary?["PLONNNBAPreseasonCacheEnabled"] as? Bool) == true
 static let sharedReportsEnabled = (Bundle.main.infoDictionary?["PLONNSharedReportsEnabled"] as? Bool) == true
 static let labURL = URL(string: "https://api.oakas.ai/plonn/v1/")!
 static let url = URL(string: "https://iwjscqdsebszqnrhbhrl.supabase.co")!
 static let publishableKey = "sb_publishable_FcAp9-EyeYGd08n9nqovkg_24XPeu2G"
}
