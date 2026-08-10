#![cfg(all(feature = "alloc", feature = "bits"))]
//! `batch_bits`: folding `DekuBitField` fields into a neighbouring bit-field run.
//!
//! The contract under test is that batching is invisible: a struct with
//! `batch_bits` must read and write exactly what the same struct without it does.

use deku::prelude::*;

/// 1-bit flag, the shape CCSDS headers are full of. Endianness comes from
/// context so the flag can sit in an `endian = "big"` struct, which is how real
/// protocol crates write these.
#[derive(Copy, Clone, Debug, PartialEq, DekuRead, DekuWrite)]
#[deku(
    id_type = "u8",
    bits = 1,
    endian = "endian",
    ctx = "endian: deku::ctx::Endian"
)]
enum Flag {
    #[deku(id = 0b0)]
    Off,
    #[deku(id = 0b1)]
    On,
}

/// 2 bits, and deliberately not every id is assigned: `0b10` must stay an error.
#[derive(Copy, Clone, Debug, PartialEq, DekuRead, DekuWrite)]
#[deku(
    id_type = "u8",
    bits = 2,
    endian = "endian",
    ctx = "endian: deku::ctx::Endian"
)]
enum Mode {
    #[deku(id = 0b00)]
    Zero,
    #[deku(id = 0b01)]
    One,
    #[deku(id = 0b11)]
    Three,
}

#[test]
fn bit_field_is_derived_for_flag_enums() {
    assert_eq!(<Flag as DekuBitField>::BITS, 1);
    assert_eq!(<Mode as DekuBitField>::BITS, 2);

    assert_eq!(Flag::from_bit_run(0b1).unwrap(), Flag::On);
    assert_eq!(Flag::On.to_bit_run().unwrap(), 0b1);

    // Bits above the field's own are not its business.
    assert_eq!(Flag::from_bit_run(0b1111_1110).unwrap(), Flag::Off);

    assert_eq!(Mode::from_bit_run(0b11).unwrap(), Mode::Three);
    assert_eq!(Mode::Three.to_bit_run().unwrap(), 0b11);
}

#[test]
fn unassigned_id_is_still_an_error() {
    assert!(Mode::from_bit_run(0b10).is_err());
}

macro_rules! header {
    ($name:ident $(, $attr:meta)?) => {
        #[derive(Debug, PartialEq, DekuRead, DekuWrite)]
        #[deku(endian = "big")]
        $(#[deku($attr)])?
        struct $name {
            #[deku(bits = 2)]
            version: u8,
            #[deku(bits = 10)]
            id: u16,
            #[deku(bits = 3)]
            channel: u8,
            ocf: Flag,
            counter: u8,
            vc_counter: u8,
            secondary: Flag,
            sync: Flag,
            order: Flag,
            mode: Mode,
            #[deku(bits = 11)]
            pointer: u16,
        }
    };
}

// Same fields, once batched and once not, so the two can be compared directly.
header!(Batched, batch_bits);
header!(Plain);

/// Every bit pattern of the 6-octet header, sampled across the whole space, must
/// decode the same batched and not, and re-encode to the input.
#[test]
fn batched_matches_unbatched() {
    let mut x: u32 = 0x1234_5678;
    for _ in 0..20_000 {
        let mut wire = [0u8; 6];
        for b in wire.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }

        let batched = Batched::from_bytes((&wire, 0));
        let plain = Plain::from_bytes((&wire, 0));

        match (batched, plain) {
            (Ok(((b_rest, b_off), b)), Ok(((p_rest, p_off), p))) => {
                assert_eq!(b_rest, p_rest);
                assert_eq!(b_off, p_off);
                assert_eq!(b.version, p.version);
                assert_eq!(b.id, p.id);
                assert_eq!(b.channel, p.channel);
                assert_eq!(b.ocf, p.ocf);
                assert_eq!(b.counter, p.counter);
                assert_eq!(b.vc_counter, p.vc_counter);
                assert_eq!(b.secondary, p.secondary);
                assert_eq!(b.sync, p.sync);
                assert_eq!(b.order, p.order);
                assert_eq!(b.mode, p.mode);
                assert_eq!(b.pointer, p.pointer);

                assert_eq!(b.to_bytes().unwrap(), wire);
                assert_eq!(p.to_bytes().unwrap(), wire);
            }
            // `Mode` rejects `0b10`, so both sides must reject the same inputs.
            (Err(_), Err(_)) => {}
            (b, p) => panic!("batched and unbatched disagree on {wire:02x?}: {b:?} vs {p:?}"),
        }
    }
}

/// A run that ends on a `DekuBitField` still leaves the cursor where the
/// unbatched reads would have, so what follows it lines up.
#[test]
fn run_followed_by_unbatchable_field() {
    #[derive(Debug, PartialEq, DekuRead, DekuWrite)]
    #[deku(endian = "big", batch_bits)]
    struct Trailing {
        #[deku(bits = 4)]
        head: u8,
        a: Flag,
        b: Flag,
        mode: Mode,
        #[deku(count = "2")]
        rest: Vec<u8>,
    }

    let wire = [0b1010_1101u8, 0xAA, 0xBB];
    let (_, v) = Trailing::from_bytes((&wire, 0)).unwrap();
    assert_eq!(
        v,
        Trailing {
            head: 0b1010,
            a: Flag::On,
            b: Flag::On,
            mode: Mode::One,
            rest: vec![0xAA, 0xBB],
        }
    );
    assert_eq!(v.to_bytes().unwrap(), wire);
}

/// A run that both starts and ends on a primitive, with the enums in the middle.
#[test]
fn enums_between_primitives() {
    #[derive(Debug, PartialEq, DekuRead, DekuWrite)]
    #[deku(endian = "big", batch_bits)]
    struct Sandwich {
        #[deku(bits = 3)]
        head: u8,
        a: Flag,
        b: Flag,
        #[deku(bits = 3)]
        tail: u8,
    }

    // head 101, a 1, b 0, tail 110
    let wire = [0b1011_0110u8];
    let (_, v) = Sandwich::from_bytes((&wire, 0)).unwrap();
    assert_eq!(
        v,
        Sandwich {
            head: 0b101,
            a: Flag::On,
            b: Flag::Off,
            tail: 0b110,
        }
    );
    assert_eq!(v.to_bytes().unwrap(), wire);
}

/// Without `batch_bits` an enum field still splits the run, which is the
/// pre-existing behaviour this must not change.
#[test]
fn without_the_attribute_nothing_changes() {
    let wire = [0xA5u8, 0x5A, 0x3C, 0xC3, 0x0F, 0xF0];
    let (_, plain) = Plain::from_bytes((&wire, 0)).unwrap();
    assert_eq!(plain.to_bytes().unwrap(), wire);
}

/// A run of enums alone, with no primitive to anchor it.
#[test]
fn all_enum_run() {
    #[derive(Debug, PartialEq, DekuRead, DekuWrite)]
    #[deku(endian = "big", batch_bits)]
    struct Flags {
        a: Flag,
        b: Flag,
        c: Flag,
        d: Flag,
        e: Flag,
        f: Flag,
        g: Flag,
        h: Flag,
    }

    let wire = [0b1100_1010u8];
    let (_, v) = Flags::from_bytes((&wire, 0)).unwrap();
    assert_eq!(
        v,
        Flags {
            a: Flag::On,
            b: Flag::On,
            c: Flag::Off,
            d: Flag::Off,
            e: Flag::On,
            f: Flag::Off,
            g: Flag::On,
            h: Flag::Off,
        }
    );
    assert_eq!(v.to_bytes().unwrap(), wire);
}

/// A wider enum id, where byte order does matter and so must be explicit.
#[test]
fn wide_id_enum() {
    #[derive(Copy, Clone, Debug, PartialEq, DekuRead, DekuWrite)]
    #[deku(id_type = "u16", bits = 12, endian = "big")]
    enum Wide {
        #[deku(id = 0x123)]
        A,
        #[deku(id = 0xABC)]
        B,
    }

    assert_eq!(<Wide as DekuBitField>::BITS, 12);

    #[derive(Debug, PartialEq, DekuRead, DekuWrite)]
    #[deku(endian = "big", batch_bits)]
    struct Holder {
        #[deku(bits = 4)]
        head: u8,
        wide: Wide,
    }

    let wire = [0x5Au8, 0xBC];
    let (_, v) = Holder::from_bytes((&wire, 0)).unwrap();
    assert_eq!(
        v,
        Holder {
            head: 0x5,
            wide: Wide::B,
        }
    );
    assert_eq!(v.to_bytes().unwrap(), wire);
}
