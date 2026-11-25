//! The JSON Schema `pattern` keyword for tool parameters, ported one to one from the pattern section of `native/src/serve_protocol.cpp`.
//!
//! A pattern is an ECMA-262 regular expression read the way JSON Schema recommends, in Unicode mode with no flags. It works on code points, `^` and `$` match only at the two ends of the string, `.` matches anything but the four line terminators, `\d`, `\w` and `\b` are ASCII, `\s` is the ECMA-262 white space set, and the pattern matches when it matches anywhere in the string. One reading goes beyond Unicode mode: a backslash before any character of the Basic Multilingual Plane that is not an ASCII letter or digit means that character. Unicode mode allows that only before syntax characters, but zod-to-json-schema escapes every literal this way.
//!
//! Refused: backreferences, named groups, Unicode property escapes, escapes that name a lone surrogate, and everything Unicode mode calls a syntax error.
//!
//! Without captures or backreferences a pattern describes a regular language, and whether a lookaround assertion holds depends on the position alone. So each lookaround body is scanned once over the whole string to tabulate where it holds, and the pattern itself is then scanned with those tables. A scan follows every thread at once and visits each step at most once per position: the time is the string length times the compiled size, whatever the pattern. No backtracking order is involved, so a lazy quantifier matches the same strings as a greedy one.

use tokspur_json::{JsonError, unicode_scalars};

use crate::admission::{AdmissionError, require};

type Result<T> = std::result::Result<T, AdmissionError>;
type Ranges = Vec<(u32, u32)>;

/// The most steps a compiled pattern may hold, over its own sequence and every lookaround body. A scan visits at most this many steps per code point.
const STEPS: u64 = 4096;
/// The deepest group nesting, which bounds the recursion of the parser and the compiler.
const DEPTH: usize = 32;
/// The most lookaround assertions. Each one costs a table of one flag per position of the matched string.
const LOOKS: usize = 16;
/// The most bytes of pattern text, which bounds what the parser builds before the step count is known.
const BYTES: usize = 16384;
const UNSUPPORTED: &str = "unsupported pattern";
const BOUND: &str = "pattern exceeds the tool pattern bound";

/// A parsed pattern. A node is `Empty` exactly when it compiles to no step.
enum Node {
    Empty,
    /// One code point of the set with this index.
    Set(usize),
    Start,
    End,
    /// A word boundary, or with `false` its absence.
    Word(bool),
    /// The lookaround body with this index must match here, or with `false` must not.
    Look(usize, bool),
    Concat(Vec<Node>),
    Either(Vec<Node>),
    /// A body repeated between a lower and an optional upper count.
    Repeat(Box<Node>, u64, Option<u64>),
}

#[derive(Clone, Copy)]
enum Step {
    Set(usize),
    Split(usize, usize),
    Jump(usize),
    Start,
    End,
    Word(bool),
    Look(usize, bool),
    Match,
}

/// One step sequence. A lookahead body is compiled in reverse and scanned from the end of the string, which finds the positions where it starts.
struct Piece {
    steps: Vec<Step>,
    backward: bool,
}

/// A compiled pattern: its character sets, then one piece per lookaround body in the order the bodies close, then the pattern's own piece.
pub(crate) struct Pattern {
    sets: Vec<Ranges>,
    pieces: Vec<Piece>,
}

fn unsupported() -> AdmissionError {
    AdmissionError::Refused(UNSUPPORTED)
}

/// The character as an ASCII byte, or zero for anything else.
fn ascii(c: u32) -> u8 {
    u8::try_from(c).ok().filter(u8::is_ascii).unwrap_or(0)
}

fn word(c: u32) -> bool {
    ascii(c).is_ascii_alphanumeric() || c == u32::from(b'_')
}

/// Sorts and merges the ranges, then takes the complement when `negate` is set.
fn normal(mut ranges: Ranges, negate: bool) -> Ranges {
    ranges.sort_unstable();
    let mut merged = Ranges::new();
    for (low, high) in ranges {
        match merged.last_mut() {
            Some(last) if low <= last.1 + 1 => last.1 = last.1.max(high),
            _ => merged.push((low, high)),
        }
    }
    if !negate {
        return merged;
    }
    let mut rest = Ranges::new();
    let mut next = 0;
    for (low, high) in merged {
        if low > next {
            rest.push((next, low - 1));
        }
        next = high + 1;
    }
    if next <= 0x10ffff {
        rest.push((next, 0x10ffff));
    }
    rest
}

/// The set a `\d`, `\w` or `\s` escape names, complemented for the capital letter.
fn shorthand(c: u32) -> Option<Ranges> {
    let ranges = match ascii(c) {
        b'd' | b'D' => vec![(0x30, 0x39)],
        b'w' | b'W' => vec![(0x30, 0x39), (0x41, 0x5a), (0x5f, 0x5f), (0x61, 0x7a)],
        // ECMA-262 WhiteSpace and LineTerminator: the ASCII controls, the byte order mark and the Unicode space separators.
        b's' | b'S' => vec![
            (0x09, 0x0d),
            (0x20, 0x20),
            (0xa0, 0xa0),
            (0x1680, 0x1680),
            (0x2000, 0x200a),
            (0x2028, 0x2029),
            (0x202f, 0x202f),
            (0x205f, 0x205f),
            (0x3000, 0x3000),
            (0xfeff, 0xfeff),
        ],
        _ => return None,
    };
    Some(normal(ranges, ascii(c).is_ascii_uppercase()))
}

struct Parser {
    source: Vec<u32>,
    at: usize,
    sets: Vec<Ranges>,
    /// Lookaround bodies in the order they close, so a body never refers to a later one. `true` marks a lookahead.
    looks: Vec<(Node, bool)>,
}

impl Parser {
    /// The next character as an ASCII byte, or zero at the end and for anything else.
    fn peek(&self) -> u8 {
        self.source.get(self.at).map_or(0, |&c| ascii(c))
    }

    fn eat(&mut self, c: u8) -> bool {
        let found = self.source.get(self.at) == Some(&u32::from(c));
        if found {
            self.at += 1;
        }
        found
    }

    fn take(&mut self) -> Result<u32> {
        let c = *self.source.get(self.at).ok_or_else(unsupported)?;
        self.at += 1;
        Ok(c)
    }

    fn set(&mut self, ranges: Ranges) -> Node {
        self.sets.push(ranges);
        Node::Set(self.sets.len() - 1)
    }

    fn disjunction(&mut self, depth: usize) -> Result<Node> {
        let mut choices = vec![self.sequence(depth)?];
        while self.eat(b'|') {
            choices.push(self.sequence(depth)?);
        }
        Ok(if choices.len() == 1 {
            choices.remove(0)
        } else {
            Node::Either(choices)
        })
    }

    fn sequence(&mut self, depth: usize) -> Result<Node> {
        let mut items = Vec::new();
        while self.at < self.source.len() && self.peek() != b'|' && self.peek() != b')' {
            let item = self.term(depth)?;
            if !matches!(item, Node::Empty) {
                items.push(item);
            }
        }
        Ok(match items.len() {
            0 => Node::Empty,
            1 => items.remove(0),
            _ => Node::Concat(items),
        })
    }

    fn term(&mut self, depth: usize) -> Result<Node> {
        let c = self.take()?;
        let atom = match ascii(c) {
            b'^' => return Ok(Node::Start),
            b'$' => return Ok(Node::End),
            b'(' => {
                require(depth < DEPTH, BOUND)?;
                let look = if self.eat(b'?') { self.group()? } else { None };
                let body = self.disjunction(depth + 1)?;
                require(self.eat(b')'), UNSUPPORTED)?;
                let Some((ahead, wanted)) = look else {
                    return self.quantified(body);
                };
                // An assertion takes no quantifier in Unicode mode.
                require(self.looks.len() < LOOKS, BOUND)?;
                self.looks.push((body, ahead));
                return Ok(Node::Look(self.looks.len() - 1, wanted));
            }
            b'[' => self.bracket()?,
            b'.' => self.set(normal(
                vec![(0x0a, 0x0a), (0x0d, 0x0d), (0x2028, 0x2029)],
                true,
            )),
            b'\\' => {
                let escaped = self.take()?;
                match ascii(escaped) {
                    b'b' => return Ok(Node::Word(true)),
                    b'B' => return Ok(Node::Word(false)),
                    _ => {}
                }
                match shorthand(escaped) {
                    Some(ranges) => self.set(ranges),
                    None => {
                        let point = self.escape(escaped, false)?;
                        self.set(vec![(point, point)])
                    }
                }
            }
            // A quantifier with nothing to repeat, or a bracket or brace that Unicode mode does not take literally.
            b'*' | b'+' | b'?' | b'{' | b'}' | b']' => return Err(unsupported()),
            _ => self.set(vec![(c, c)]),
        };
        self.quantified(atom)
    }

    /// What follows `(?`: nothing to record for a plain group, otherwise whether the assertion looks ahead and whether its body must match.
    fn group(&mut self) -> Result<Option<(bool, bool)>> {
        Ok(match ascii(self.take()?) {
            b':' => None,
            b'=' => Some((true, true)),
            b'!' => Some((true, false)),
            b'<' => match ascii(self.take()?) {
                b'=' => Some((false, true)),
                b'!' => Some((false, false)),
                _ => return Err(unsupported()),
            },
            _ => return Err(unsupported()),
        })
    }

    fn quantified(&mut self, atom: Node) -> Result<Node> {
        let (low, high) = match self.peek() {
            b'*' => (0, None),
            b'+' => (1, None),
            b'?' => (0, Some(1)),
            b'{' => {
                self.at += 1;
                let low = self.count()?;
                let high = if !self.eat(b',') {
                    Some(low)
                } else if self.peek().is_ascii_digit() {
                    Some(self.count()?)
                } else {
                    None
                };
                require(
                    self.peek() == b'}' && high.is_none_or(|high| low <= high),
                    UNSUPPORTED,
                )?;
                (low, high)
            }
            _ => return Ok(atom),
        };
        self.at += 1;
        // A lazy quantifier changes which match is found first, never whether one exists.
        self.eat(b'?');
        Ok(if matches!(atom, Node::Empty) || high == Some(0) {
            Node::Empty
        } else {
            Node::Repeat(Box::new(atom), low, high)
        })
    }

    fn count(&mut self) -> Result<u64> {
        let start = self.at;
        let mut value = 0;
        while self.peek().is_ascii_digit() {
            value = (value * 10 + u64::from(self.peek() - b'0')).min(STEPS + 1);
            self.at += 1;
        }
        require(self.at > start, UNSUPPORTED)?;
        require(value <= STEPS, BOUND)?;
        Ok(value)
    }

    fn bracket(&mut self) -> Result<Node> {
        let negate = self.eat(b'^');
        let mut ranges = Ranges::new();
        loop {
            let c = self.take()?;
            if c == u32::from(b']') {
                break;
            }
            let low = self.member(c, &mut ranges)?;
            // A dash is literal at either end of the class and makes a range anywhere else.
            if self.peek() == b'-'
                && self
                    .source
                    .get(self.at + 1)
                    .is_some_and(|&next| next != u32::from(b']'))
            {
                self.at += 1;
                let c = self.take()?;
                let high = self.member(c, &mut ranges)?;
                let (Some(low), Some(high)) = (low, high) else {
                    return Err(unsupported());
                };
                require(low <= high, UNSUPPORTED)?;
                ranges.push((low, high));
            } else if let Some(low) = low {
                ranges.push((low, low));
            }
        }
        Ok(self.set(normal(ranges, negate)))
    }

    /// One member of a class: its code point, or `None` after adding the ranges of a shorthand set, which cannot end a range.
    fn member(&mut self, c: u32, ranges: &mut Ranges) -> Result<Option<u32>> {
        if c != u32::from(b'\\') {
            return Ok(Some(c));
        }
        let escaped = self.take()?;
        if let Some(set) = shorthand(escaped) {
            ranges.extend(set);
            return Ok(None);
        }
        self.escape(escaped, true).map(Some)
    }

    /// The code point an escape names, given the character after the backslash.
    fn escape(&mut self, c: u32, bracketed: bool) -> Result<u32> {
        Ok(match ascii(c) {
            b't' => 0x09,
            b'n' => 0x0a,
            b'v' => 0x0b,
            b'f' => 0x0c,
            b'r' => 0x0d,
            b'b' if bracketed => 0x08,
            b'0' => {
                require(!self.peek().is_ascii_digit(), UNSUPPORTED)?;
                0
            }
            b'c' => {
                let letter = self.take()?;
                require(ascii(letter).is_ascii_alphabetic(), UNSUPPORTED)?;
                letter % 32
            }
            b'x' => self.hex(2)?,
            b'u' => {
                let mut point = 0;
                if self.eat(b'{') {
                    let mut digits = 0;
                    while !self.eat(b'}') {
                        point = point * 16 + self.hex(1)?;
                        require(point <= 0x10ffff, UNSUPPORTED)?;
                        digits += 1;
                    }
                    require(digits > 0, UNSUPPORTED)?;
                } else {
                    point = self.hex(4)?;
                    if (0xd800..0xdc00).contains(&point) {
                        // Two escapes that form a surrogate pair name one code point.
                        require(self.eat(b'\\') && self.eat(b'u'), UNSUPPORTED)?;
                        let trail = self.hex(4)?;
                        require((0xdc00..0xe000).contains(&trail), UNSUPPORTED)?;
                        point = 0x10000 + ((point - 0xd800) << 10) + (trail - 0xdc00);
                    }
                }
                // A surrogate on its own never occurs in a string of scalars.
                require(!(0xd800..0xe000).contains(&point), UNSUPPORTED)?;
                point
            }
            // An ASCII letter or digit reaching here is a backreference, a property escape or no escape at all. Any other character of the Basic Multilingual Plane stands for itself. Beyond that plane engines that take such an escape disagree on what it means.
            other => {
                require(!other.is_ascii_alphanumeric() && c < 0x10000, UNSUPPORTED)?;
                c
            }
        })
    }

    fn hex(&mut self, count: usize) -> Result<u32> {
        let mut value = 0;
        for _ in 0..count {
            let digit = char::from(ascii(self.take()?))
                .to_digit(16)
                .ok_or_else(unsupported)?;
            value = value * 16 + digit;
        }
        Ok(value)
    }
}

/// The steps a node compiles to, clamped just above the bound so that no sum overflows.
fn size(node: &Node) -> u64 {
    let total = match node {
        Node::Empty => 0,
        Node::Set(_) | Node::Start | Node::End | Node::Word(_) | Node::Look(..) => 1,
        Node::Concat(items) => items
            .iter()
            .fold(0, |total, item| (total + size(item)).min(STEPS + 1)),
        Node::Either(items) => items
            .iter()
            .fold(2 * (items.len() as u64 - 1), |total, item| {
                (total + size(item)).min(STEPS + 1)
            }),
        Node::Repeat(body, low, high) => {
            let body = size(body);
            match high {
                Some(high) => low * body + (high - low) * (body + 1),
                None if *low == 0 => body + 2,
                None => low * body + 1,
            }
        }
    };
    total.min(STEPS + 1)
}

/// Appends the node's steps. A backward piece consumes the string from its end, so a sequence is laid out last item first.
fn emit(node: &Node, backward: bool, steps: &mut Vec<Step>) {
    match node {
        Node::Empty => {}
        Node::Set(set) => steps.push(Step::Set(*set)),
        Node::Start => steps.push(Step::Start),
        Node::End => steps.push(Step::End),
        Node::Word(wanted) => steps.push(Step::Word(*wanted)),
        Node::Look(piece, wanted) => steps.push(Step::Look(*piece, *wanted)),
        Node::Concat(items) if backward => {
            for item in items.iter().rev() {
                emit(item, backward, steps);
            }
        }
        Node::Concat(items) => {
            for item in items {
                emit(item, backward, steps);
            }
        }
        Node::Either(items) => {
            let mut exits = Vec::new();
            for (index, item) in items.iter().enumerate() {
                if index + 1 == items.len() {
                    emit(item, backward, steps);
                    break;
                }
                let split = steps.len();
                steps.push(Step::Split(split + 1, 0));
                emit(item, backward, steps);
                exits.push(steps.len());
                steps.push(Step::Jump(0));
                steps[split] = Step::Split(split + 1, steps.len());
            }
            for exit in exits {
                steps[exit] = Step::Jump(steps.len());
            }
        }
        Node::Repeat(body, low, high) => {
            let mut last = steps.len();
            for _ in 0..*low {
                last = steps.len();
                emit(body, backward, steps);
            }
            match high {
                Some(high) => {
                    let mut splits = Vec::new();
                    for _ in *low..*high {
                        splits.push(steps.len());
                        steps.push(Step::Split(0, 0));
                        emit(body, backward, steps);
                    }
                    for split in splits {
                        steps[split] = Step::Split(split + 1, steps.len());
                    }
                }
                None if *low == 0 => {
                    let split = steps.len();
                    steps.push(Step::Split(0, 0));
                    emit(body, backward, steps);
                    steps.push(Step::Jump(split));
                    steps[split] = Step::Split(split + 1, steps.len());
                }
                None => steps.push(Step::Split(last, steps.len() + 1)),
            }
        }
    }
}

/// Parses and compiles a pattern, refusing one outside the supported grammar or over a bound.
pub(crate) fn compile(source: &[u8]) -> Result<Pattern> {
    require(source.len() <= BYTES, BOUND)?;
    let source = unicode_scalars(source).map_err(JsonError::from)?;
    let mut parser = Parser {
        source,
        at: 0,
        sets: Vec::new(),
        looks: Vec::new(),
    };
    let root = parser.disjunction(0)?;
    // Only an unmatched closing parenthesis stops the parser early.
    require(parser.at == parser.source.len(), UNSUPPORTED)?;
    // The pattern itself is the last piece and scans forward, as a lookbehind body does.
    parser.looks.push((root, false));
    let total = parser.looks.iter().fold(0, |total, (body, _)| {
        (total + size(body) + 1).min(STEPS + 1)
    });
    require(total <= STEPS, BOUND)?;
    let pieces = parser
        .looks
        .iter()
        .map(|(body, backward)| {
            let mut steps = Vec::new();
            emit(body, *backward, &mut steps);
            steps.push(Step::Match);
            Piece {
                steps,
                backward: *backward,
            }
        })
        .collect();
    Ok(Pattern {
        sets: parser.sets,
        pieces,
    })
}

impl Pattern {
    /// Whether the pattern matches anywhere in the string, given as code points.
    pub(crate) fn matches(&self, text: &[u32]) -> bool {
        let mut tables: Vec<Vec<bool>> = Vec::with_capacity(self.pieces.len());
        for (index, piece) in self.pieces.iter().enumerate() {
            // A lookaround table needs every position. The pattern's own piece comes last and needs one match.
            let hits = self.scan(piece, text, &tables, index + 1 == self.pieces.len());
            tables.push(hits);
        }
        tables.last().is_some_and(|hits| hits.contains(&true))
    }

    /// The positions at which a match of the piece ends, for a backward piece the positions at which one starts. A thread starts at every position. With `once` set the scan stops at its first match.
    fn scan(&self, piece: &Piece, text: &[u32], tables: &[Vec<bool>], once: bool) -> Vec<bool> {
        let mut hits = vec![false; text.len() + 1];
        // The scan pass in which each step was last entered, so a step is entered once per position however many threads reach it.
        let mut entered = vec![0; piece.steps.len()];
        let mut stack = Vec::new();
        let mut waiting = Vec::new();
        for pass in 1..=text.len() + 1 {
            let at = if piece.backward {
                text.len() + 1 - pass
            } else {
                pass - 1
            };
            stack.push(0);
            waiting.clear();
            while let Some(step) = stack.pop() {
                if entered[step] == pass {
                    continue;
                }
                entered[step] = pass;
                let proceed = match piece.steps[step] {
                    Step::Set(_) => {
                        waiting.push(step);
                        false
                    }
                    Step::Split(first, second) => {
                        stack.push(first);
                        stack.push(second);
                        false
                    }
                    Step::Jump(target) => {
                        stack.push(target);
                        false
                    }
                    Step::Start => at == 0,
                    Step::End => at == text.len(),
                    Step::Word(wanted) => {
                        let before = at > 0 && word(text[at - 1]);
                        let after = at < text.len() && word(text[at]);
                        (before != after) == wanted
                    }
                    Step::Look(table, wanted) => tables[table][at] == wanted,
                    Step::Match => {
                        hits[at] = true;
                        false
                    }
                };
                if proceed {
                    stack.push(step + 1);
                }
            }
            if pass > text.len() || (once && hits[at]) {
                break;
            }
            let c = if piece.backward {
                text[at - 1]
            } else {
                text[at]
            };
            // The stack is empty here, so it carries the threads that consumed this character into the next position.
            for &step in &waiting {
                if let Step::Set(set) = piece.steps[step] {
                    let ranges = &self.sets[set];
                    let index = ranges.partition_point(|&(_, high)| high < c);
                    if ranges.get(index).is_some_and(|&(low, _)| low <= c) {
                        stack.push(step + 1);
                    }
                }
            }
        }
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal text of compiling a pattern, or empty when it is admitted.
    fn refusal(pattern: &str) -> String {
        match compile(pattern.as_bytes()) {
            Ok(_) => String::new(),
            Err(error) => error.to_string(),
        }
    }

    fn found(pattern: &str, text: &str) -> bool {
        let scalars: Vec<u32> = text.chars().map(u32::from).collect();
        compile(pattern.as_bytes()).unwrap().matches(&scalars)
    }

    /// Checks that the pattern matches every string of `yes` and no string of `no`.
    fn sorts(pattern: &str, yes: &[&str], no: &[&str]) {
        for text in yes {
            assert!(found(pattern, text), "{pattern} should match {text:?}");
        }
        for text in no {
            assert!(!found(pattern, text), "{pattern} should not match {text:?}");
        }
    }

    #[test]
    fn a_pattern_matches_anywhere_unless_anchored() {
        sorts("b", &["b", "abc", "ab"], &["", "a", "B"]);
        sorts("^b", &["b", "bc"], &["ab", "a\nb"]);
        sorts("b$", &["b", "ab"], &["ba", "b\n"]);
        sorts("^$", &[""], &["a", "\n"]);
        sorts("", &["", "anything"], &[]);
        sorts("^a|b$", &["ax", "xb"], &["xa", "bx"]);
        sorts("a^b", &[], &["ab", "a\nb"]);
        sorts("a$b", &[], &["ab", "a\nb"]);
    }

    #[test]
    fn classes_take_ranges_negation_and_literal_dashes() {
        sorts("^[a-c]$", &["a", "b", "c"], &["d", "-", "ab"]);
        sorts("^[^a-c]$", &["d", "\n", "\u{1f600}"], &["a", "c", ""]);
        sorts("^[-a]$", &["-", "a"], &["b"]);
        sorts("^[a-]$", &["-", "a"], &["b"]);
        sorts("^[a\\-c]$", &["a", "-", "c"], &["b"]);
        sorts("^[+--]$", &["+", ",", "-"], &["."]);
        sorts("^[a-c-e]$", &["a", "b", "c", "-", "e"], &["d"]);
        sorts("^[A-Za-z0-9-_]$", &["Q", "q", "5", "-", "_"], &["+"]);
        sorts("^[\\d-]$", &["7", "-"], &["a"]);
        sorts("^[\\w.]$", &["a", "_", "."], &["-"]);
        sorts("^[^\\s-]$", &["a"], &[" ", "-", "\t"]);
        sorts("^[\\b]$", &["\u{8}"], &["b"]);
        sorts(
            "^[.*+?(){}|^$\\[\\]\\\\]$",
            &[".", "*", "{", "^", "[", "]", "\\"],
            &["a"],
        );
        sorts("^[]$", &[], &["", "a"]);
        sorts("^[^]$", &["a", "\n", "\u{10ffff}"], &["", "ab"]);
        for pattern in [
            "[", "[a", "[a-", "[b-a]", "[\\d-z]", "[a-\\w]", "[a--]", "a]",
        ] {
            assert_eq!(refusal(pattern), UNSUPPORTED, "{pattern}");
        }
    }

    #[test]
    fn shorthand_sets_and_the_dot_are_the_ecma_262_ones() {
        sorts("^\\d$", &["0", "9"], &["\u{663}", "a", "\u{ff11}"]);
        sorts("^\\D$", &["\u{663}", "a", "\n"], &["0", "9"]);
        sorts(
            "^\\w$",
            &["a", "Z", "0", "_"],
            &["\u{e9}", "-", "\u{17f}", "\u{212a}"],
        );
        sorts("^\\W$", &["\u{e9}", "-", " "], &["a", "_"]);
        let spaces = [
            "\t", "\n", "\u{b}", "\u{c}", "\r", " ", "\u{a0}", "\u{1680}", "\u{2000}", "\u{200a}",
            "\u{2028}", "\u{2029}", "\u{202f}", "\u{205f}", "\u{3000}", "\u{feff}",
        ];
        let others = ["\u{85}", "\u{180e}", "\u{200b}", "\u{2060}", "a", "\u{1c}"];
        sorts("^\\s$", &spaces, &others);
        sorts("^\\S$", &others, &spaces);
        sorts(
            "^.$",
            &["a", "\u{85}", "\t", "\u{b}", "\u{1f600}"],
            &["\n", "\r", "\u{2028}", "\u{2029}", ""],
        );
    }

    #[test]
    fn a_character_is_one_code_point() {
        sorts(
            "^.$",
            &["\u{1f600}", "\u{e9}"],
            &["e\u{301}", "\u{1f600}\u{1f600}"],
        );
        sorts("^.{2}$", &["e\u{301}", "\u{1f600}a"], &["\u{1f600}"]);
        sorts(
            "^\u{1f600}+$",
            &["\u{1f600}", "\u{1f600}\u{1f600}"],
            &["", "\u{1f601}"],
        );
        sorts(
            "^[\u{1f600}-\u{1f64f}]$",
            &["\u{1f600}", "\u{1f64f}"],
            &["\u{1f650}", "a"],
        );
        sorts("^\\u{1F600}$", &["\u{1f600}"], &["a"]);
        sorts("^\\uD83D\\uDE00$", &["\u{1f600}"], &["a"]);
        sorts(
            "^\\u00e9\\xe9\\u{0000e9}$",
            &["\u{e9}\u{e9}\u{e9}"],
            &["eee"],
        );
        sorts(
            "^\\t\\n\\v\\f\\r\\0\\cJ\\cj$",
            &["\t\n\u{b}\u{c}\r\0\n\n"],
            &[],
        );
        for pattern in [
            "\\uD83D",
            "\\uDE00",
            "\\uD83Dx",
            "\\uD83D\\u0041",
            "\\u{D800}",
            "\\u{110000}",
            "\\u{}",
            "\\u{1F600",
            "\\u12",
            "\\x1",
            "\\xg0",
            "\\c1",
            "\\c",
            "\\00",
            "\\01",
        ] {
            assert_eq!(refusal(pattern), UNSUPPORTED, "{pattern}");
        }
    }

    #[test]
    fn quantifiers_groups_and_alternation() {
        sorts("^a*$", &["", "a", "aaa"], &["b", "ab"]);
        sorts("^a+?$", &["a", "aaa"], &[""]);
        sorts("^a??b$", &["b", "ab"], &["aab"]);
        sorts("^a{2}$", &["aa"], &["a", "aaa"]);
        sorts("^a{2,}$", &["aa", "aaaaa"], &["a"]);
        sorts("^a{1,3}?$", &["a", "aa", "aaa"], &["", "aaaa"]);
        sorts("^a{0}b$", &["b"], &["ab"]);
        sorts("^a{0,0}$", &[""], &["a"]);
        sorts("^(ab|c)+$", &["ab", "c", "abcab"], &["a", "ac", ""]);
        sorts("^(?:a|)b$", &["ab", "b"], &["aab"]);
        sorts("^(|a|)$", &["", "a"], &["aa"]);
        sorts("^()*a(?:)+$", &["a"], &["", "aa"]);
        sorts("^(a*)*$", &["", "aaaa"], &["b"]);
        sorts("^(a*)+b$", &["b", "aab"], &["a"]);
        sorts(
            "^(a|ab)(c|bcd)(d*)$",
            &["abcd", "ac", "abc", "acdd"],
            &["ab", "abd"],
        );
        sorts("^((a)|(b))*?c$", &["c", "abbac"], &["abd"]);
        for pattern in [
            "*",
            "+a",
            "?",
            "a**",
            "a+*",
            "a*??",
            "a{",
            "a{1",
            "a{,2}",
            "a{1,2",
            "a{2,1}",
            "a{x}",
            "{1}",
            "a}",
            "^*",
            "$+",
            "\\b?",
            "(?=a)*",
            "(?<!a){2}",
            "(",
            ")",
            "(a",
            "a)",
            "(?",
            "(?a)",
            "(?i)a",
            "(?<n>a)",
            "(?P<n>a)",
            "(?<",
            "(?#c)",
        ] {
            assert_eq!(refusal(pattern), UNSUPPORTED, "{pattern}");
        }
    }

    #[test]
    fn word_boundaries_are_ascii() {
        sorts(
            "\\bcat\\b",
            &["cat", "a cat.", "cat-flap"],
            &["cats", "concat", "\u{e9}cat_"],
        );
        sorts("\\Bcat", &["concat", "_cat"], &["cat", " cat", "\u{e9}cat"]);
        sorts("^\\b", &["a"], &["", "-a", "\u{e9}"]);
        sorts("\\b$", &["a", "-a"], &["", "a-"]);
        sorts("^\\B$", &[""], &["a"]);
        sorts(
            "^[0-9a-fA-F]{8}\\b-[0-9a-fA-F]{4}\\b-",
            &["01234567-89ab-"],
            &["01234567_89ab-"],
        );
    }

    #[test]
    fn lookaround_holds_by_position() {
        sorts(
            "^(?=.*\\d)(?=.*[a-z]).{4,}$",
            &["ab12", "1a.."],
            &["abcd", "1234", "a1"],
        );
        sorts("^(?!ab)\\w+$", &["ba", "a", "aab"], &["ab", "abc"]);
        sorts("a(?=b)", &["ab", "xab"], &["a", "ac", "ba"]);
        sorts("a(?!b)", &["a", "ac", "aba"], &["ab", "xab"]);
        sorts("(?<=a)b", &["ab", "xab"], &["b", "cb"]);
        sorts("(?<!a)b", &["b", "cb", "abb"], &["ab", "xab"]);
        sorts("^(?=a$)a$", &["a"], &["aa"]);
        sorts("^a(?<=^a)$", &["a"], &[]);
        sorts("^(?:(?=a)\\w|-)+$", &["a-a", "-", "aa"], &["ab", "b"]);
        sorts("^(?=(?!ab)a)\\w+$", &["ac", "a"], &["ab", "b"]);
        sorts("^(?=a(?<=^a)b)ab$", &["ab"], &[]);
        sorts("(?<=(?<!x)a)b", &["ab"], &["xab", "b"]);
        sorts("^(?!)", &[], &["", "a"]);
        sorts("^(?=)$", &[""], &["a"]);
        sorts("(?=\\b)a", &["a", " a"], &["ba"]);
        sorts("^(?=(a|b)*c)\\w+$", &["abc", "c", "cab"], &["abd"]);
        // zod's email, hostname and ISO duration patterns, and Pydantic's Decimal pattern.
        sorts(
            "^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$",
            &["a@b.co", "first.last+tag@sub.example.org", "o'neil@x-y.io"],
            &[
                ".a@b.co",
                "a..b@c.de",
                "a.@b.co",
                "a@b",
                "a@-b.co",
                "a b@c.de",
                "a@b.c",
            ],
        );
        sorts(
            "^(?=.{1,253}\\.?$)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[-0-9a-zA-Z]{0,61}[0-9a-zA-Z])?)*\\.?$",
            &["example.com", "a", "x-1.y.", "localhost"],
            &["", "-a.com", "a-.com", "a..b", "a_b.com"],
        );
        sorts(
            "^P(?:(\\d+W)|(?!.*W)(?=\\d|T\\d)(\\d+Y)?(\\d+M)?(\\d+D)?(T(?=\\d)(\\d+H)?(\\d+M)?(\\d+([.,]\\d+)?S)?)?)$",
            &["P1W", "P1Y2M3D", "PT1H", "P1DT2H3M4.5S", "PT0,5S"],
            &["P", "PT", "P1WT1H", "P1S", "1Y", "P1Y2W"],
        );
        sorts(
            "^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$",
            &["1", "-1.5", "+.5", "007", "1."],
            &["", "-", "+.", ".", "1.2.3", "1e5", "--1"],
        );
        sorts(
            "^(?!^[-+.]*$)[+-]?0*(?:\\d{0,3}|(?=[\\d.]{1,6}0*$)\\d{0,3}\\.\\d{0,2}0*$)",
            &["123", "123.45", "-0.5", "1.50"],
            &["-", ".", "+"],
        );
    }

    #[test]
    fn a_backslash_before_a_non_alphanumeric_character_is_that_character() {
        sorts(
            "^https\\:\\/\\/a\\ b\\_c\\-d\\.$",
            &["https://a b_c-d."],
            &["https://a b_c-dx"],
        );
        sorts(
            "^\\@\\#\\%\\&\\,\\;\\<\\=\\>\\~\\`\\!\\'\\\"$",
            &["@#%&,;<=>~`!'\""],
            &[],
        );
        sorts(
            "^na\\\u{ef}ve\\\u{2028}$",
            &["na\u{ef}ve\u{2028}"],
            &["naive"],
        );
        assert_eq!(refusal("\\\u{1f600}"), UNSUPPORTED);
        assert_eq!(refusal("[\\\u{1f600}]"), UNSUPPORTED);
        sorts("^[\\ -\\~]+$", &["a b~"], &["\t", "\u{e9}"]);
        sorts(
            "^\\^\\$\\\\\\.\\*\\+\\?\\(\\)\\[\\]\\{\\}\\|$",
            &["^$\\.*+?()[]{}|"],
            &[],
        );
        for pattern in [
            "\\a", "\\e", "\\g", "\\h", "\\A", "\\Z", "\\z", "\\G", "\\R", "\\X", "\\Q", "\\1",
            "(a)\\1", "\\9", "\\k<n>", "\\p{L}", "\\P{Lu}", "[\\p{L}]", "[\\B]", "[\\1]", "[\\c1]",
            "\\",
        ] {
            assert_eq!(refusal(pattern), UNSUPPORTED, "{pattern}");
        }
    }

    #[test]
    fn generator_patterns_without_lookaround_are_admitted() {
        // zod 4's own JSON Schema output, then zod-to-json-schema and Pydantic.
        for pattern in [
            "^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|00000000-0000-0000-0000-000000000000|ffffffff-ffff-ffff-ffff-ffffffffffff)$",
            "^[a-zA-Z0-9_-]{21}$",
            "^[cC][^\\s-]{8,}$",
            "^[0-9A-HJKMNP-TV-Za-hjkmnp-tv-z]{26}$",
            "^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$",
            "^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:))$",
            "^$|^(?:[0-9a-zA-Z+/]{4})*(?:(?:[0-9a-zA-Z+/]{2}==)|(?:[0-9a-zA-Z+/]{3}=))?$",
            "^\\+(?:[0-9]){6,14}[0-9]$",
            "^(?:(?:\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12]\\d|3[01])|(?:0[469]|11)-(?:0[1-9]|[12]\\d|30)|(?:02)-(?:0[1-9]|1\\d|2[0-8])))T(?:(?:[01]\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d(?:\\.\\d+)?)?(?:Z|([+-](?:[01]\\d|2[0-3]):[0-5]\\d)))$",
            "^https:\\/\\/a b_c-d\\..*",
            "^id--?\\d+(?:\\.\\d+)?-(a|b)$",
            "^[^A-Z]*$",
            "^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$",
            "^([0-9a-zA-Z-_]{4})*(([0-9a-zA-Z-_]{2}(==)?)|([0-9a-zA-Z-_]{3}(=)?))?$",
            "^[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]*$",
            "^\\s*(\\d*\\.?\\d+)\\s*(\\w+)?",
        ] {
            assert_eq!(refusal(pattern), "", "{pattern}");
        }
        assert_eq!(
            refusal("^(\\p{Extended_Pictographic}|\\p{Emoji_Component})+$"),
            UNSUPPORTED
        );
    }

    /// The step count the size check works with, and the steps the compiler lays out.
    fn steps(pattern: &str) -> (u64, usize) {
        let mut parser = Parser {
            source: unicode_scalars(pattern.as_bytes()).unwrap(),
            at: 0,
            sets: Vec::new(),
            looks: Vec::new(),
        };
        let root = parser.disjunction(0).unwrap();
        parser.looks.push((root, false));
        let counted = parser.looks.iter().map(|(body, _)| size(body) + 1).sum();
        let compiled = compile(pattern.as_bytes()).unwrap();
        let laid = compiled.pieces.iter().map(|piece| piece.steps.len()).sum();
        (counted, laid)
    }

    #[test]
    fn the_size_check_counts_the_steps_the_compiler_lays_out() {
        for (pattern, expected) in [
            ("", 1),
            ("a", 2),
            ("ab", 3),
            ("a|b", 5),
            ("a|b|", 7),
            ("a*", 4),
            ("a+", 3),
            ("a?", 3),
            ("a{3}", 4),
            ("a{2,}", 4),
            ("a{2,4}", 7),
            ("(?:ab){2,}c", 7),
            ("(?:a|b)*", 7),
            ("()*", 1),
            ("a{0}", 1),
            ("(?:a{0})+b", 2),
            ("^\\ba$", 5),
            ("(?=a)b", 5),
            ("(?:(?<!ab)c){2}", 8),
            ("(?:(?:(?=a)){0})b", 4),
        ] {
            assert_eq!(steps(pattern), (expected, expected as usize), "{pattern}");
        }
    }

    #[test]
    fn a_pattern_over_a_bound_is_refused() {
        assert_eq!(refusal("a{4095}"), "");
        assert_eq!(refusal("a{4096}"), BOUND);
        assert_eq!(refusal("(?:a{63}){65}"), "");
        assert_eq!(refusal("(?:a{64}){64}"), BOUND);
        assert_eq!(refusal("a{0,2047}"), "");
        assert_eq!(refusal("a{0,2048}"), BOUND);
        assert_eq!(refusal("(?:(?:(?:a{20}){20}){20}){20}"), BOUND);
        assert_eq!(refusal("(?:(?:(?:a*)*)*)*"), "");
        assert_eq!(refusal(&"a".repeat(4095)), "");
        assert_eq!(refusal(&"a".repeat(4096)), BOUND);
        assert_eq!(refusal(&"|".repeat(2047)), "");
        assert_eq!(refusal(&"|".repeat(2048)), BOUND);
        // A count above the bound is refused whatever it repeats, and before its order is looked at.
        assert_eq!(refusal("a{4096}"), BOUND);
        assert_eq!(refusal("(){4097}"), BOUND);
        assert_eq!(refusal("a{5000,4000}"), BOUND);
        assert_eq!(refusal("a{99999999999999999999999999}"), BOUND);
        assert_eq!(refusal("(){4096}"), "");
        assert_eq!(refusal("a{4096,}"), BOUND);
        let nested = |depth: usize| format!("{}a{}", "(".repeat(depth), ")".repeat(depth));
        assert_eq!(refusal(&nested(32)), "");
        assert_eq!(refusal(&nested(33)), BOUND);
        assert_eq!(refusal(&format!("{}a", "(?=".repeat(33))), BOUND);
        assert_eq!(refusal(&"(?=a)".repeat(16)), "");
        assert_eq!(refusal(&"(?=a)".repeat(17)), BOUND);
        assert_eq!(refusal(&"(?:(?:(?=a)){0})".repeat(17)), BOUND);
    }

    #[test]
    fn nested_quantifiers_take_linear_time() {
        // Each of these takes a backtracking matcher longer than the test would wait.
        let text = "a".repeat(20_000);
        assert!(!found("^(a*)*b$", &text));
        assert!(!found("^(a|a)*b$", &text));
        assert!(!found("^(a+)+$", &format!("{text}b")));
        assert!(!found("(.*a){12}b", &text));
        assert!(found("^(?=(a*)*$)(a|aa)+$", &text));
        assert!(!found("^(?!(a+)+$)", &text));
        assert!(found("(?<=(a*)*)b", &format!("{text}b")));
    }
}
