use crate::pristine::*;

pub const START_MARKER: &str = ">>>>>>>";

pub const SEPARATOR: &str = "=======";

pub const END_MARKER: &str = "<<<<<<<";

/// A trait for outputting keys and their contents. This trait allows
/// to retain more information about conflicts than directly
/// outputting as bytes to a `Write`. The diff algorithm uses that
/// information, for example.
pub trait VertexBuffer {
    fn output_line<E, F>(&mut self, key: Vertex<ChangeId>, contents: F) -> Result<(), E>
    where
        E: From<std::io::Error>,
        F: FnOnce(&mut [u8]) -> Result<(), E>;

    fn output_conflict_marker(
        &mut self,
        s: &str,
        id: usize,
        sides: &[&Hash],
    ) -> Result<(), std::io::Error>;
    fn begin_conflict(&mut self, id: usize, side: &[&Hash]) -> Result<(), std::io::Error> {
        self.output_conflict_marker(START_MARKER, id, side)
    }
    fn begin_zombie_conflict(
        &mut self,
        id: usize,
        add_del: &[&Hash],
    ) -> Result<(), std::io::Error> {
        self.output_conflict_marker(START_MARKER, id, add_del)
    }
    fn begin_cyclic_conflict(&mut self, id: usize) -> Result<(), std::io::Error> {
        self.output_conflict_marker(START_MARKER, id, &[])
    }
    fn conflict_next(&mut self, id: usize, side: &[&Hash]) -> Result<(), std::io::Error> {
        self.output_conflict_marker(SEPARATOR, id, side)
    }
    fn end_conflict(&mut self, id: usize) -> Result<(), std::io::Error> {
        self.output_conflict_marker(END_MARKER, id, &[])
    }
    fn end_zombie_conflict(&mut self, id: usize) -> Result<(), std::io::Error> {
        self.end_conflict(id)
    }
    fn end_cyclic_conflict(&mut self, id: usize) -> Result<(), std::io::Error> {
        self.output_conflict_marker(END_MARKER, id, &[])
    }
}

pub(crate) struct ConflictsWriter<'a, 'b, W: std::io::Write> {
    pub w: W,
    pub lines: usize,
    pub new_line: bool,
    pub path: &'b str,
    pub inode_vertex: Position<ChangeId>,
    pub conflicts: &'a mut Vec<crate::output::Conflict>,
    pub buf: Vec<u8>,
}

impl<'a, 'b, W: std::io::Write> ConflictsWriter<'a, 'b, W> {
    pub fn new(
        w: W,
        path: &'b str,
        inode_vertex: Position<ChangeId>,
        conflicts: &'a mut Vec<crate::output::Conflict>,
    ) -> Self {
        ConflictsWriter {
            inode_vertex,
            w,
            new_line: true,
            lines: 1,
            path,
            conflicts,
            buf: Vec::new(),
        }
    }
}

impl<'a, 'b, W: std::io::Write> std::ops::Deref for ConflictsWriter<'a, 'b, W> {
    type Target = W;
    fn deref(&self) -> &Self::Target {
        &self.w
    }
}

impl<'a, 'b, W: std::io::Write> std::ops::DerefMut for ConflictsWriter<'a, 'b, W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.w
    }
}

impl<'a, 'b, W: std::io::Write> VertexBuffer for ConflictsWriter<'a, 'b, W> {
    fn output_line<E, C>(&mut self, v: Vertex<ChangeId>, c: C) -> Result<(), E>
    where
        E: From<std::io::Error>,
        C: FnOnce(&mut [u8]) -> Result<(), E>,
    {
        self.buf.resize(v.end - v.start, 0);
        c(&mut self.buf)?;
        debug!("vbuf {:?} {:?}", v, std::str::from_utf8(&self.buf));
        let ends_with_newline = self.buf.ends_with(b"\n");
        self.lines += self.buf.iter().filter(|c| **c == b'\n').count();
        self.w.write_all(&self.buf)?;
        if !self.buf.is_empty() {
            // empty "lines" (such as in the beginning of a file)
            // don't change the status of self.new_line.
            self.new_line = ends_with_newline;
        }
        Ok(())
    }

    fn output_conflict_marker(
        &mut self,
        s: &str,
        id: usize,
        sides: &[&Hash],
    ) -> Result<(), std::io::Error> {
        debug!("output_conflict_marker {:?}", self.new_line);
        if !self.new_line {
            self.lines += 2;
            self.w.write_all(b"\n")?;
        } else {
            self.lines += 1;
            debug!("{:?}", s.as_bytes());
        }
        write!(self.w, "{} {}", s, id)?;
        for side in sides {
            let h = side.to_base32();
            write!(self.w, " [{}]", h.split_at(8).0)?;
        }
        self.w.write_all(b"\n")?;
        self.new_line = true;
        Ok(())
    }

    fn begin_conflict(&mut self, id: usize, side: &[&Hash]) -> Result<(), std::io::Error> {
        self.conflicts.push(crate::output::Conflict::Order {
            path: self.path.to_string(),
            inode_vertex: [self.inode_vertex],
            line: self.lines,
            changes: side.iter().cloned().cloned().collect(),
            id,
        });
        self.output_conflict_marker(START_MARKER, id, side)
    }
    fn begin_zombie_conflict(
        &mut self,
        id: usize,
        add_del: &[&Hash],
    ) -> Result<(), std::io::Error> {
        self.conflicts.push(crate::output::Conflict::Zombie {
            path: self.path.to_string(),
            inode_vertex: [self.inode_vertex],
            line: self.lines,
            changes: add_del.iter().cloned().cloned().collect(),
            id,
        });
        self.output_conflict_marker(START_MARKER, id, add_del)
    }
    fn begin_cyclic_conflict(&mut self, id: usize) -> Result<(), std::io::Error> {
        self.conflicts.push(crate::output::Conflict::Cyclic {
            path: self.path.to_string(),
            inode_vertex: [self.inode_vertex],
            line: self.lines,
            changes: Vec::new(),
            id,
        });
        self.output_conflict_marker(START_MARKER, id, &[])
    }
    fn conflict_next(&mut self, id_: usize, side: &[&Hash]) -> Result<(), std::io::Error> {
        for conflict in self.conflicts.iter_mut().rev() {
            match conflict {
                crate::output::Conflict::Order { id, changes, .. } if *id == id_ => {
                    changes.extend(side.into_iter().cloned())
                }
                crate::output::Conflict::Zombie { id, changes, .. } if *id == id_ => {
                    changes.extend(side.into_iter().cloned())
                }
                crate::output::Conflict::Cyclic { id, changes, .. } if *id == id_ => {
                    changes.extend(side.into_iter().cloned())
                }
                _ => break,
            }
        }
        self.output_conflict_marker(SEPARATOR, id_, side)
    }
}

pub struct Writer<W: std::io::Write> {
    w: W,
    buf: Vec<u8>,
    new_line: bool,
    is_zombie: bool,
}

impl<W: std::io::Write> Writer<W> {
    pub fn new(w: W) -> Self {
        Writer {
            w,
            new_line: true,
            buf: Vec::new(),
            is_zombie: false,
        }
    }
    pub fn into_inner(self) -> W {
        self.w
    }
}

impl<W: std::io::Write> std::ops::Deref for Writer<W> {
    type Target = W;
    fn deref(&self) -> &Self::Target {
        &self.w
    }
}

impl<W: std::io::Write> std::ops::DerefMut for Writer<W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.w
    }
}

impl<W: std::io::Write> VertexBuffer for Writer<W> {
    fn output_line<E, C>(&mut self, v: Vertex<ChangeId>, c: C) -> Result<(), E>
    where
        E: From<std::io::Error>,
        C: FnOnce(&mut [u8]) -> Result<(), E>,
    {
        self.buf.resize(v.end - v.start, 0);
        c(&mut self.buf)?;
        debug!("vbuf {:?} {:?}", v, std::str::from_utf8(&self.buf));
        let ends_with_newline = self.buf.ends_with(b"\n");
        self.w.write_all(&self.buf[..])?;
        if !self.buf.is_empty() {
            // empty "lines" (such as in the beginning of a file)
            // don't change the status of self.new_line.
            self.new_line = ends_with_newline;
        }
        Ok(())
    }

    fn output_conflict_marker(
        &mut self,
        s: &str,
        id: usize,
        sides: &[&Hash],
    ) -> Result<(), std::io::Error> {
        debug!("output_conflict_marker {:?}", self.new_line);
        if !self.new_line {
            self.w.write_all(b"\n")?;
        }
        write!(self.w, "{} {}", s, id)?;
        for side in sides {
            let h = side.to_base32();
            write!(self.w, " [{}]", h.split_at(8).0)?;
        }
        self.w.write_all(b"\n")?;
        Ok(())
    }

    fn begin_conflict(&mut self, id: usize, side: &[&Hash]) -> Result<(), std::io::Error> {
        self.output_conflict_marker(START_MARKER, id, side)
    }
    fn end_conflict(&mut self, id: usize) -> Result<(), std::io::Error> {
        self.is_zombie = false;
        self.output_conflict_marker(END_MARKER, id, &[])
    }
    fn begin_zombie_conflict(
        &mut self,
        id: usize,
        add_del: &[&Hash],
    ) -> Result<(), std::io::Error> {
        if self.is_zombie {
            Ok(())
        } else {
            self.is_zombie = true;
            self.output_conflict_marker(START_MARKER, id, add_del)
        }
    }
    fn end_zombie_conflict(&mut self, id: usize) -> Result<(), std::io::Error> {
        self.is_zombie = false;
        self.output_conflict_marker(END_MARKER, id, &[])
    }
    fn begin_cyclic_conflict(&mut self, id: usize) -> Result<(), std::io::Error> {
        self.output_conflict_marker(START_MARKER, id, &[])
    }
}
