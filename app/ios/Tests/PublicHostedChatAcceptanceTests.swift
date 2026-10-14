import XCTest
@testable import PLONNCore

final class PublicHostedChatAcceptanceTests: XCTestCase {
    func testPublicGatewayRejectsNoncredentialThroughActualClientTransport() async throws {
        guard ProcessInfo.processInfo.environment["PLONN_RUN_PUBLIC_CHAT_AUTH_CHECK"] == "1" else {
            throw XCTSkip("Opt-in harmless public auth rejection; no session, inference or paid provider use.")
        }
        let provider = HostedAskPLONNProvider(tokenProvider: { "noncredential-plonn-beta-auth-check" })
        let request = AskPLONNRequest(question: "Explain implied probability in general terms.",
                                      intent: .generalChat, evidence: [])
        do {
            _ = try await provider.answer(request)
            XCTFail("The public service accepted a noncredential")
        } catch let error as HostedAskPLONNError {
            XCTAssertEqual(error, .authenticationRequired)
            print("PUBLIC_CHAT_CLIENT_AUTH_REJECTED: actual HTTPS POST returned authenticationRequired; no issued session used.")
        }
    }
}
