import Foundation

/// A decoded JSON value of unknown shape.
///
/// Several protocol fields are deliberately open — a job's `payload`, a tool
/// call's `arguments`, a result's `observed`, a personal read's `items`. Their
/// shape belongs to the workload, not to the envelope, so the envelope refuses
/// to guess at it: it carries the value faithfully and lets the layer that
/// knows the workload interpret it.
///
/// Modelling them as `[String: Any]` would have cost `Codable` conformance and
/// with it the round-trip guarantee the fixtures exist to prove.
public enum JSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "unsupported JSON value"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }

    // MARK: - Convenience

    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    public var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    public var arrayValue: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }

    public var doubleValue: Double? {
        if case .number(let value) = self { return value }
        return nil
    }

    /// The value as an `Int`, or nil if it is not a whole number in range.
    ///
    /// JSON has one number type, so a field a schema calls an integer arrives
    /// here as a `Double`. Converting through `Int(exactly:)` rather than
    /// truncating means a fractional or out-of-range value reads as absent
    /// instead of silently becoming a different number.
    public var intValue: Int? {
        guard case .number(let value) = self else { return nil }
        return Int(exactly: value.rounded())
    }

    public subscript(key: String) -> JSONValue? {
        objectValue?[key]
    }
}
