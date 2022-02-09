use crate::{ParallelParam, ParallelVec};
use alloc::vec::Vec;
use serde::{
    de::DeserializeOwned, ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer,
};

impl<'a, Param> Serialize for ParallelVec<Param>
where
    Param: ParallelParam + 'a,
    Param::Ref<'a>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for item in self.iter() {
            seq.serialize_element(&item)?;
        }
        seq.end()
    }
}

impl<'de, Param> Deserialize<'de> for ParallelVec<Param>
where
    Param: ParallelParam + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::from(<Vec<Param> as Deserialize<'de>>::deserialize(
            deserializer,
        )?))
    }
}

#[cfg(test)]
mod test {
    use crate::ParallelVec;
    use serde_test::{assert_tokens, Token};

    #[test]
    fn test_serde_empty() {
        let vec: ParallelVec<(u64, i32)> = ParallelVec::new();
        assert_tokens(&vec, &[Token::Seq { len: Some(0) }, Token::SeqEnd]);
    }

    #[test]
    fn test_serde_2() {
        let vec: ParallelVec<(i32, u64)> = ParallelVec::from(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_tokens(
            &vec,
            &[
                Token::Seq { len: Some(4) },
                Token::Tuple { len: 2 },
                Token::I32(1),
                Token::U64(2),
                Token::TupleEnd,
                Token::Tuple { len: 2 },
                Token::I32(3),
                Token::U64(4),
                Token::TupleEnd,
                Token::Tuple { len: 2 },
                Token::I32(5),
                Token::U64(6),
                Token::TupleEnd,
                Token::Tuple { len: 2 },
                Token::I32(7),
                Token::U64(8),
                Token::TupleEnd,
                Token::SeqEnd,
            ],
        );
    }

    #[test]
    fn test_serde_3() {
        let vec: ParallelVec<(i32, u64, f32)> =
            ParallelVec::from(vec![(1, 2, 0.0), (3, 4, -1.0), (5, 6, -2.0), (7, 8, -3.0)]);
        assert_tokens(
            &vec,
            &[
                Token::Seq { len: Some(4) },
                Token::Tuple { len: 3 },
                Token::I32(1),
                Token::U64(2),
                Token::F32(0.0),
                Token::TupleEnd,
                Token::Tuple { len: 3 },
                Token::I32(3),
                Token::U64(4),
                Token::F32(-1.0),
                Token::TupleEnd,
                Token::Tuple { len: 3 },
                Token::I32(5),
                Token::U64(6),
                Token::F32(-2.0),
                Token::TupleEnd,
                Token::Tuple { len: 3 },
                Token::I32(7),
                Token::U64(8),
                Token::F32(-3.0),
                Token::TupleEnd,
                Token::SeqEnd,
            ],
        );
    }
}
