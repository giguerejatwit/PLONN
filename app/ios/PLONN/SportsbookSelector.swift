import SwiftUI

/// An inline sportsbook picker. The caller owns the available names and selection.
struct SportsbookSelector: View {
    @Binding var selection: String
    let options: [String]

    @State private var isExpanded = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @AccessibilityFocusState private var headerFocused: Bool

    // Preserve caller order while avoiding duplicate SwiftUI row identities.
    private var uniqueOptions: [String] {
        var seen = Set<String>()
        return options.filter { seen.insert($0).inserted }
    }

    private var selectionLabel: String {
        selection.isEmpty ? "Choose sportsbook" : selection
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                setExpanded(!isExpanded)
            } label: {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Sportsbook")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Text(selectionLabel)
                            .font(.system(.subheadline, design: .monospaced).weight(.semibold))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 8)
                    Image(systemName: "chevron.down")
                        .font(.system(.caption, design: .monospaced).weight(.semibold))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                        .accessibilityHidden(true)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                .contentShape(Rectangle())
                .plonnGlassSurface(radius: 12)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Sportsbook")
            .accessibilityValue("\(selectionLabel), \(isExpanded ? "expanded" : "collapsed")")
            .accessibilityHint(options.isEmpty ? "No sportsbooks available" : isExpanded ? "Hide sportsbook choices" : "Show sportsbook choices")
            .accessibilityFocused($headerFocused)
            .disabled(options.isEmpty)

            if isExpanded && !options.isEmpty {
                VStack(spacing: 2) {
                    ForEach(uniqueOptions, id: \.self) { book in
                        optionRow(book)
                    }
                }
                .padding(4)
                .background(navySurface, in: RoundedRectangle(cornerRadius: 12))
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(Color.primary.opacity(0.12))
                        .allowsHitTesting(false)
                }
                .transition(.opacity)
            }
        }
        .foregroundStyle(.primary)
        .onChange(of: options) { _, updated in
            if updated.isEmpty { setExpanded(false) }
        }
        .accessibilityAction(.escape) {
            setExpanded(false)
            headerFocused = true
        }
    }

    private func optionRow(_ book: String) -> some View {
        Button {
            selection = book
            setExpanded(false)
            headerFocused = true
        } label: {
            HStack(spacing: 12) {
                Text(book)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                Image(systemName: "checkmark")
                    .fontWeight(.semibold)
                    .opacity(selection == book ? 1 : 0)
                    .accessibilityHidden(true)
            }
            .font(.system(.subheadline, design: .monospaced))
            .padding(.horizontal, 10)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            .background(selection == book ? Color.primary.opacity(0.08) : Color.clear,
                        in: RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(book)
        .accessibilityAddTraits(selection == book ? .isSelected : [])
        .accessibilityHint("Select sportsbook and close choices")
    }

    private func setExpanded(_ expanded: Bool) {
        withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.18)) {
            isExpanded = expanded
        }
    }
}
